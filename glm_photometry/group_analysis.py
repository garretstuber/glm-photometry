"""
Full-group circular-shift permutation GLM for fiber photometry.

Runs the GLM per subject, then compares encoding (delta-R2) across
the three VTA GABA populations (Cbln4, Crhbp, Pnoc).

Elerding et al. -- VTA GABA neuron subtypes
"""

import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from scipy import stats as sp_stats
import time
import warnings
warnings.filterwarnings("ignore")

from .core import (
    get_trial_matrix, build_design_matrix, run_permutation_glm,
    run_timeresolved, prepare_channel_data, svd_setup,
    compute_fstats_1d, circular_shift_1d,
    N_SHIFTS, ALPHA, REFERENCE_SOLUTION,
)


# ====================================================================
# H5 LOADER
# ====================================================================

def load_h5_data(h5_path):
    """Load the full-group h5 file into a pandas DataFrame."""
    with h5py.File(h5_path, "r") as f:
        grp = f["streams_peth"]
        data = {}
        for key in grp.keys():
            arr = grp[key][:]
            if arr.dtype.kind == "S":
                arr = np.array([x.decode() for x in arr])
            data[key] = arr
    df = pd.DataFrame(data)
    df = df.sort_values(
        ["channel_id", "subject", "blockname", "solution",
         "event_number_solution", "time_rel"]
    ).reset_index(drop=True)
    return df


# ====================================================================
# PER-SUBJECT GLM
# ====================================================================

def run_subject_glm(df, subject, n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
    """
    Run full-trial and time-resolved GLM for a single subject.

    The subject's data is treated as if it were a single-channel dataset
    (which it is -- one fiber per subject). We group by blockname (session).
    """
    df_subj = df[df["subject"] == subject].copy()
    channel_id = df_subj["channel_id"].iloc[0]
    sessions = sorted(df_subj["blockname"].unique())

    # Prepare data (reuse prepare_channel_data logic but on subject-filtered df)
    # We temporarily set a unique channel_id for this subject to isolate it
    Y_parts = []
    trial_info_list = []
    session_lengths = []
    session_trial_counts = []
    time_rel = None
    T_trial = None

    for sess in sessions:
        df_sess = df_subj[df_subj["blockname"] == sess].copy()
        Y_sess, ti_sess, tr, Tt, nt = get_trial_matrix(df_sess)
        Y_parts.append(Y_sess)
        trial_info_list.append(ti_sess)
        session_lengths.append(len(Y_sess))
        session_trial_counts.append(nt)
        if time_rel is None:
            time_rel = tr
            T_trial = Tt

    Y = np.concatenate(Y_parts)
    boundaries = []
    offset = 0
    for sl in session_lengths:
        boundaries.append((offset, offset + sl))
        offset += sl

    solutions = sorted(df["solution"].unique())
    X_list, chunk_names = build_design_matrix(
        trial_info_list, T_trial, session_lengths, solutions=solutions
    )

    # Full-trial GLM
    full_res = run_permutation_glm(
        Y, X_list, boundaries,
        n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=False,
    )

    # Time-resolved GLM
    tr_res = run_timeresolved(
        Y, trial_info_list, time_rel, T_trial,
        session_trial_counts,
        win_size=1.0, win_step=0.25,
        n_shifts=n_shifts, alpha=alpha, seed=seed,
        verbose=False,
    )

    return {
        "subject": subject,
        "channel_id": channel_id,
        "n_sessions": len(sessions),
        "n_trials": sum(session_trial_counts),
        "full_glm": full_res,
        "timeresolved": tr_res,
        "chunk_names": chunk_names,
        "time_rel": time_rel,
    }


# ====================================================================
# GROUP-LEVEL STATISTICS
# ====================================================================

def run_group_statistics(all_subject_results, alpha=0.05):
    """
    Compare delta-R2 across channel groups for each predictor.

    Tests:
      - Kruskal-Wallis H test (non-parametric one-way ANOVA)
      - Pairwise Mann-Whitney U post-hoc tests
      - Proportion of significant subjects per group (chi-squared)

    Returns
    -------
    stats_df : DataFrame with test results
    """
    channels = sorted(set(r["channel_id"] for r in all_subject_results))
    chunk_names = all_subject_results[0]["chunk_names"]
    # Exclude session (nuisance)
    task_predictors = [n for n in chunk_names if n != "session"]

    rows = []
    for pred_idx, pred_name in enumerate(chunk_names):
        if pred_name == "session":
            continue

        # Collect delta-R2 per group
        groups = {ch: [] for ch in channels}
        sig_groups = {ch: [] for ch in channels}
        for r in all_subject_results:
            ch = r["channel_id"]
            groups[ch].append(r["full_glm"]["delta_r2_obs"][pred_idx])
            sig_groups[ch].append(r["full_glm"]["significant"][pred_idx])

        # Kruskal-Wallis
        arrays = [np.array(groups[ch]) for ch in channels]
        if all(len(a) > 0 for a in arrays):
            kw_stat, kw_p = sp_stats.kruskal(*arrays)
        else:
            kw_stat, kw_p = np.nan, np.nan

        # Group means and SEMs
        for ch in channels:
            arr = np.array(groups[ch])
            sig_arr = np.array(sig_groups[ch])
            rows.append({
                "predictor": pred_name,
                "channel_id": ch,
                "n": len(arr),
                "mean_dr2": np.mean(arr),
                "sem_dr2": np.std(arr) / np.sqrt(len(arr)) if len(arr) > 1 else 0,
                "median_dr2": np.median(arr),
                "n_sig": int(np.sum(sig_arr)),
                "pct_sig": np.mean(sig_arr) * 100,
                "kw_stat": kw_stat,
                "kw_p": kw_p,
            })

        # Pairwise Mann-Whitney
        for i, ch1 in enumerate(channels):
            for ch2 in channels[i+1:]:
                a1 = np.array(groups[ch1])
                a2 = np.array(groups[ch2])
                if len(a1) > 1 and len(a2) > 1:
                    mw_stat, mw_p = sp_stats.mannwhitneyu(
                        a1, a2, alternative="two-sided"
                    )
                else:
                    mw_stat, mw_p = np.nan, np.nan
                rows.append({
                    "predictor": pred_name,
                    "channel_id": f"{ch1}_vs_{ch2}",
                    "n": len(a1) + len(a2),
                    "mean_dr2": np.nan,
                    "sem_dr2": np.nan,
                    "median_dr2": np.nan,
                    "n_sig": np.nan,
                    "pct_sig": np.nan,
                    "kw_stat": np.nan,
                    "kw_p": np.nan,
                    "mw_stat": mw_stat,
                    "mw_p": mw_p,
                })

    stats_df = pd.DataFrame(rows)
    if "mw_stat" not in stats_df.columns:
        stats_df["mw_stat"] = np.nan
        stats_df["mw_p"] = np.nan
    return stats_df


def run_timeresolved_group_stats(all_subject_results):
    """
    Aggregate time-resolved delta-R2 across subjects within each channel group.

    Returns
    -------
    tr_group : dict mapping channel_id -> {
        win_centers, mean_dr2 (n_windows, K), sem_dr2, pct_sig
    }
    """
    channels = sorted(set(r["channel_id"] for r in all_subject_results))
    chunk_names = all_subject_results[0]["timeresolved"]["chunk_names"]
    win_centers = all_subject_results[0]["timeresolved"]["win_centers"]
    n_win = len(win_centers)
    K = len(chunk_names)

    tr_group = {}
    for ch in channels:
        subj_results = [r for r in all_subject_results if r["channel_id"] == ch]
        n_subj = len(subj_results)
        dr2_stack = np.zeros((n_subj, n_win, K))
        sig_stack = np.zeros((n_subj, n_win, K), dtype=bool)

        for si, r in enumerate(subj_results):
            tr = r["timeresolved"]
            n_w = min(n_win, tr["delta_r2"].shape[0])
            dr2_stack[si, :n_w, :] = tr["delta_r2"][:n_w, :]
            sig_stack[si, :n_w, :] = tr["significant"][:n_w, :]

        tr_group[ch] = {
            "win_centers": win_centers,
            "mean_dr2": np.mean(dr2_stack, axis=0),
            "sem_dr2": np.std(dr2_stack, axis=0) / np.sqrt(n_subj),
            "pct_sig": np.mean(sig_stack, axis=0) * 100,
            "n_subj": n_subj,
            "chunk_names": chunk_names,
        }

    return tr_group


# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_group_results(all_subject_results, stats_df, tr_group, save_dir):
    """Generate all group-comparison figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        matplotlib.font_manager.fontManager.addfont(
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
        )
        plt.rcParams["font.family"] = "Liberation Sans"
    except Exception:
        plt.rcParams["font.family"] = "sans-serif"

    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    save_dir = Path(save_dir)
    channels = sorted(set(r["channel_id"] for r in all_subject_results))
    chunk_names = all_subject_results[0]["chunk_names"]
    task_preds = [i for i, n in enumerate(chunk_names) if n != "session"]
    task_names = [chunk_names[i] for i in task_preds]
    ch_colors = {"cbln4": "#4C72B0", "crhbp": "#DD8452", "pnoc": "#55A868"}
    pred_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    # ── Figure 1: Per-subject delta-R2 swarm + bar ──
    n_pred = len(task_preds)
    fig, axes = plt.subplots(1, n_pred, figsize=(2.8 * n_pred, 3.5), sharey=False)
    if n_pred == 1:
        axes = [axes]

    for pi, (pred_idx, pred_name) in enumerate(zip(task_preds, task_names)):
        ax = axes[pi]
        x_positions = np.arange(len(channels))

        for ci, ch in enumerate(channels):
            vals = [r["full_glm"]["delta_r2_obs"][pred_idx]
                    for r in all_subject_results if r["channel_id"] == ch]
            vals = np.array(vals)
            mean = np.mean(vals)
            sem = np.std(vals) / np.sqrt(len(vals))

            # Bar
            ax.bar(ci, mean, width=0.55, color=ch_colors[ch],
                   edgecolor="black", linewidth=0.5, alpha=0.7)
            # Error bar
            ax.errorbar(ci, mean, yerr=sem, color="black", capsize=3,
                        linewidth=0.8, capthick=0.8)
            # Scatter individual subjects
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(ci + jitter, vals, s=12, color=ch_colors[ch],
                       edgecolor="black", linewidth=0.3, zorder=3, alpha=0.8)

        # Add Kruskal-Wallis p-value
        kw_row = stats_df[
            (stats_df["predictor"] == pred_name)
            & (~stats_df["channel_id"].str.contains("_vs_"))
        ]
        if len(kw_row) > 0:
            kw_p = kw_row.iloc[0]["kw_p"]
            if kw_p < 0.001:
                p_str = "p < 0.001"
            elif kw_p < 0.05:
                p_str = f"p = {kw_p:.3f}"
            else:
                p_str = f"p = {kw_p:.2f}"
            ax.set_title(f"{pred_name}\nKW: {p_str}", fontsize=9)

        # Pairwise significance brackets
        y_max = ax.get_ylim()[1]
        bracket_y = y_max * 1.0
        pw_rows = stats_df[
            (stats_df["predictor"] == pred_name)
            & (stats_df["channel_id"].str.contains("_vs_"))
        ]
        for _, pw in pw_rows.iterrows():
            if pd.notna(pw.get("mw_p")) and pw["mw_p"] < 0.05:
                pair = pw["channel_id"].split("_vs_")
                i1 = channels.index(pair[0])
                i2 = channels.index(pair[1])
                bracket_y += y_max * 0.08
                ax.plot([i1, i1, i2, i2],
                        [bracket_y - y_max*0.02, bracket_y,
                         bracket_y, bracket_y - y_max*0.02],
                        color="black", linewidth=0.7)
                star = "***" if pw["mw_p"] < 0.001 else (
                    "**" if pw["mw_p"] < 0.01 else "*")
                ax.text((i1 + i2) / 2, bracket_y + y_max * 0.01, star,
                        ha="center", fontsize=8)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(channels, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("$\\Delta R^2$")

    fig.suptitle("Per-subject encoding strength across VTA GABA populations",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "group_dr2_comparison.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: Proportion significant per group ──
    fig, ax = plt.subplots(figsize=(5, 3.5))
    bar_width = 0.22
    x = np.arange(n_pred)

    for ci, ch in enumerate(channels):
        pcts = []
        for pred_name in task_names:
            row = stats_df[
                (stats_df["predictor"] == pred_name)
                & (stats_df["channel_id"] == ch)
            ]
            pcts.append(row.iloc[0]["pct_sig"] if len(row) > 0 else 0)
        offset = (ci - 1) * bar_width
        ax.bar(x + offset, pcts, width=bar_width, color=ch_colors[ch],
               edgecolor="black", linewidth=0.5, label=ch, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, fontsize=8)
    ax.set_ylabel("% subjects significant\n(p < 0.01)")
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Proportion of subjects with significant encoding",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_dir / "group_pct_significant.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Time-resolved group means +/- SEM ──
    n_plot_preds = min(4, len(task_preds))
    fig, axes = plt.subplots(len(channels), 1,
                             figsize=(5.5, 2.4 * len(channels)),
                             sharex=True, squeeze=False)

    for ci, ch in enumerate(channels):
        ax = axes[ci, 0]
        tg = tr_group[ch]
        for pi in range(n_plot_preds):
            k = task_preds[pi]
            mean = tg["mean_dr2"][:, k]
            sem = tg["sem_dr2"][:, k]
            label = tg["chunk_names"][k]
            ax.plot(tg["win_centers"], mean, color=pred_colors[pi],
                    linewidth=1.2, label=label)
            ax.fill_between(tg["win_centers"], mean - sem, mean + sem,
                            color=pred_colors[pi], alpha=0.15)

        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel("$\\Delta R^2$\n(mean +/- SEM)")
        ax.set_title(f"{ch} (n = {tg['n_subj']})",
                     fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ci == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)

    axes[-1, 0].set_xlabel("Time from access onset (s)")
    fig.suptitle("Time-resolved encoding: group means",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_dir / "group_timeresolved.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 4: Time-resolved overlay (all groups on one axis per predictor) ──
    fig, axes = plt.subplots(1, n_plot_preds,
                             figsize=(3.0 * n_plot_preds, 3.0),
                             sharey=False)
    if n_plot_preds == 1:
        axes = [axes]

    for pi in range(n_plot_preds):
        ax = axes[pi]
        k = task_preds[pi]
        for ci, ch in enumerate(channels):
            tg = tr_group[ch]
            mean = tg["mean_dr2"][:, k]
            sem = tg["sem_dr2"][:, k]
            ax.plot(tg["win_centers"], mean, color=ch_colors[ch],
                    linewidth=1.3, label=ch)
            ax.fill_between(tg["win_centers"], mean - sem, mean + sem,
                            color=ch_colors[ch], alpha=0.12)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(task_names[pi], fontsize=9, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("$\\Delta R^2$")
            ax.legend(fontsize=7, frameon=False)
        ax.set_xlabel("Time (s)")

    fig.suptitle("Time-resolved encoding: group comparison",
                 fontsize=11, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(save_dir / "group_timeresolved_overlay.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 5: Full model R2 per group ──
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for ci, ch in enumerate(channels):
        vals = [r["full_glm"]["r2_full"]
                for r in all_subject_results if r["channel_id"] == ch]
        vals = np.array(vals)
        mean = np.mean(vals)
        sem = np.std(vals) / np.sqrt(len(vals))
        ax.bar(ci, mean, width=0.55, color=ch_colors[ch],
               edgecolor="black", linewidth=0.5, alpha=0.7)
        ax.errorbar(ci, mean, yerr=sem, color="black", capsize=3,
                    linewidth=0.8, capthick=0.8)
        jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(ci + jitter, vals, s=12, color=ch_colors[ch],
                   edgecolor="black", linewidth=0.3, zorder=3, alpha=0.8)

    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, fontsize=9)
    ax.set_ylabel("Full model $R^2$")
    ax.set_title("Total variance explained", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(save_dir / "group_r2_full.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {save_dir}")


# ====================================================================
# MAIN
# ====================================================================

def main(h5_path, save_dir=None, n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
    h5_path = Path(h5_path)
    if save_dir is None:
        save_dir = h5_path.parent
    save_dir = Path(save_dir)

    print("Loading data...")
    df = load_h5_data(h5_path)
    subjects = sorted(df["subject"].unique())
    channels = sorted(df["channel_id"].unique())
    print(f"Subjects: {len(subjects)}")
    print(f"Channels: {channels}")
    print(f"N per channel: {df.groupby('channel_id')['subject'].nunique().to_dict()}")
    print(f"Total rows: {len(df):,}")
    print(f"Permutations per subject: {n_shifts}")

    # ── Per-subject GLM ──
    all_results = []
    t0 = time.time()
    for si, subj in enumerate(subjects):
        ch = df[df["subject"] == subj]["channel_id"].iloc[0]
        t_subj = time.time()
        print(f"\n[{si+1}/{len(subjects)}] {subj} ({ch})...", end=" ", flush=True)

        res = run_subject_glm(df, subj, n_shifts=n_shifts, alpha=alpha,
                              seed=seed + si)
        all_results.append(res)

        elapsed = time.time() - t_subj
        sig_preds = [res["chunk_names"][k] for k in range(len(res["chunk_names"]))
                     if res["full_glm"]["significant"][k] and
                     res["chunk_names"][k] != "session"]
        print(f"{elapsed:.1f}s  R2={res['full_glm']['r2_full']:.3f}  "
              f"sig: {sig_preds if sig_preds else 'none'}")

    total_time = time.time() - t0
    print(f"\n\nAll subjects completed in {total_time:.0f}s ({total_time/60:.1f} min)")

    # ── Group statistics ──
    print("\nRunning group statistics...")
    stats_df = run_group_statistics(all_results, alpha=0.05)

    # Print summary
    print(f"\n{'='*70}")
    print("GROUP COMPARISON SUMMARY")
    print(f"{'='*70}")
    chunk_names = all_results[0]["chunk_names"]
    task_preds = [n for n in chunk_names if n != "session"]

    for pred in task_preds:
        print(f"\n--- {pred} ---")
        pred_rows = stats_df[
            (stats_df["predictor"] == pred)
            & (~stats_df["channel_id"].str.contains("_vs_"))
        ]
        for _, row in pred_rows.iterrows():
            print(f"  {row['channel_id']}: mean dR2={row['mean_dr2']:.4f} "
                  f"+/- {row['sem_dr2']:.4f}  "
                  f"({row['n_sig']:.0f}/{row['n']:.0f} sig = {row['pct_sig']:.0f}%)")
        kw_p = pred_rows.iloc[0]["kw_p"]
        print(f"  Kruskal-Wallis: H={pred_rows.iloc[0]['kw_stat']:.2f}, p={kw_p:.4f}")

        # Pairwise
        pw_rows = stats_df[
            (stats_df["predictor"] == pred)
            & (stats_df["channel_id"].str.contains("_vs_"))
        ]
        for _, pw in pw_rows.iterrows():
            if pd.notna(pw.get("mw_p")):
                sig = " *" if pw["mw_p"] < 0.05 else ""
                print(f"  {pw['channel_id']}: U={pw['mw_stat']:.1f}, "
                      f"p={pw['mw_p']:.4f}{sig}")

    # ── Time-resolved group stats ──
    tr_group = run_timeresolved_group_stats(all_results)

    # ── Figures ──
    print("\nGenerating figures...")
    plot_group_results(all_results, stats_df, tr_group, save_dir)

    # ── Save results ──
    # Per-subject results table
    subj_rows = []
    for r in all_results:
        for k, name in enumerate(r["chunk_names"]):
            subj_rows.append({
                "subject": r["subject"],
                "channel_id": r["channel_id"],
                "n_sessions": r["n_sessions"],
                "n_trials": r["n_trials"],
                "predictor": name,
                "delta_r2": r["full_glm"]["delta_r2_obs"][k],
                "f_obs": r["full_glm"]["f_obs"][k],
                "p_value": r["full_glm"]["pvals"][k],
                "significant": r["full_glm"]["significant"][k],
                "r2_full_model": r["full_glm"]["r2_full"],
            })
    subj_df = pd.DataFrame(subj_rows)
    subj_df.to_csv(save_dir / "glm_per_subject_results.csv", index=False)
    stats_df.to_csv(save_dir / "glm_group_statistics.csv", index=False)

    print(f"\nResults saved:")
    print(f"  {save_dir / 'glm_per_subject_results.csv'}")
    print(f"  {save_dir / 'glm_group_statistics.csv'}")

    return all_results, stats_df, tr_group


if __name__ == "__main__":
    main("streams_peth_multi_mix_all.h5")
