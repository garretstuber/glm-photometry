"""
Valence encoding and timing analyses for VTA GABA neuron subtypes.

Analysis 1 -- Appetitive vs. aversive encoding:
  Categorize solutions as appetitive (sucrose30), aversive (nacl150, dry),
  or neutral (water). Compute mean peri-event response per category per
  subject, then derive a valence index and compare across populations.

Analysis 2 -- Encoding onset latency:
  Use existing time-resolved GLM results to extract the earliest window
  with significant encoding for each predictor per subject. Compare
  latencies across populations, with focus on Crhbp vs Pnoc.

Garret Stuber lab, UW -- Elerding et al., VTA GABA neurons
"""

import sys, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, ttest_1samp, wilcoxon

sys.path.insert(0, str(Path("mnt/20260331 - Elerding et al., VTA GABA neurons")))
from fp_glm_group_analysis import load_h5_data

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

SAVE_DIR = Path("mnt/20260331 - Elerding et al., VTA GABA neurons")
H5 = SAVE_DIR / "streams_peth_multi_mix_all.h5"
SIGNAL_COL = "delta_signal_poly_zscore_blsub"

POP_MAP = {"cbln4": "Cbln4", "crhbp": "Crhbp", "pnoc": "Pnoc"}
POP_COLORS = {"Cbln4": "#4C72B0", "Crhbp": "#DD8452", "Pnoc": "#55A868"}

# Valence categories
APPETITIVE = ["sucrose30"]
AVERSIVE = ["nacl150", "dry"]
NEUTRAL = ["water"]

# Response window (seconds after access onset)
RESP_WIN = (0.0, 3.0)   # primary response window
BL_WIN = (-2.0, -0.5)   # baseline window


# ====================================================================
# ANALYSIS 1: VALENCE ENCODING
# ====================================================================

def compute_valence_metrics(df):
    """
    For each subject, compute:
      - Mean response to appetitive, aversive, neutral solutions
      - Valence index: (appetitive - aversive) / (|appetitive| + |aversive|)
      - Appetitive selectivity: appetitive - neutral
      - Aversive selectivity: aversive - neutral
    """
    subjects = sorted(df["subject"].unique())
    rows = []

    for subj in subjects:
        df_subj = df[df["subject"] == subj]
        channel_id = df_subj["channel_id"].iloc[0]
        pop = POP_MAP.get(channel_id, channel_id)

        # Get time axis
        time_rel = df_subj["time_rel"].values
        resp_mask = (time_rel >= RESP_WIN[0]) & (time_rel < RESP_WIN[1])
        bl_mask = (time_rel >= BL_WIN[0]) & (time_rel < BL_WIN[1])

        # Compute mean baseline-subtracted response per solution
        sol_means = {}
        sol_traces = {}
        for sol in ["water", "sucrose30", "nacl150", "dry"]:
            df_sol = df_subj[df_subj["solution"] == sol]
            if len(df_sol) == 0:
                continue

            # Get all trials for this solution
            trials = df_sol.groupby(["blockname", "event_number_solution"])
            trial_resps = []
            trial_traces = []
            for _, trial_df in trials:
                sig = trial_df[SIGNAL_COL].values
                tr = trial_df["time_rel"].values
                r_mask = (tr >= RESP_WIN[0]) & (tr < RESP_WIN[1])
                b_mask = (tr >= BL_WIN[0]) & (tr < BL_WIN[1])
                if r_mask.sum() > 0 and b_mask.sum() > 0:
                    bl = np.nanmean(sig[b_mask])
                    resp = np.nanmean(sig[r_mask])
                    trial_resps.append(resp - bl)
                    trial_traces.append(sig)

            if trial_resps:
                sol_means[sol] = np.nanmean(trial_resps)
                sol_traces[sol] = np.nanmean(trial_traces, axis=0)

        if not all(s in sol_means for s in ["water", "sucrose30", "nacl150", "dry"]):
            continue

        app_resp = sol_means["sucrose30"]
        avers_resp = np.mean([sol_means["nacl150"], sol_means["dry"]])
        neut_resp = sol_means["water"]

        # Valence index: positive = appetitive-preferring, negative = aversive-preferring
        denom = abs(app_resp) + abs(avers_resp)
        valence_idx = (app_resp - avers_resp) / denom if denom > 1e-10 else 0.0

        rows.append({
            "subject": subj,
            "channel_id": channel_id,
            "population": pop,
            "resp_water": sol_means["water"],
            "resp_sucrose": sol_means["sucrose30"],
            "resp_nacl": sol_means["nacl150"],
            "resp_dry": sol_means["dry"],
            "resp_appetitive": app_resp,
            "resp_aversive": avers_resp,
            "resp_neutral": neut_resp,
            "valence_index": valence_idx,
            "app_selectivity": app_resp - neut_resp,
            "avers_selectivity": avers_resp - neut_resp,
        })

    return pd.DataFrame(rows)


def compute_population_traces(df):
    """Compute mean +/- SEM traces per solution per population."""
    subjects = sorted(df["subject"].unique())
    time_rel = np.sort(df["time_rel"].unique())

    # Collect per-subject mean traces
    pop_sol_traces = {}  # (pop, sol) -> list of (179,) arrays
    for subj in subjects:
        df_subj = df[df["subject"] == subj]
        channel_id = df_subj["channel_id"].iloc[0]
        pop = POP_MAP.get(channel_id, channel_id)

        for sol in ["water", "sucrose30", "nacl150", "dry"]:
            df_sol = df_subj[df_subj["solution"] == sol]
            trials = df_sol.groupby(["blockname", "event_number_solution"])
            trial_sigs = []
            for _, trial_df in trials:
                trial_df_sorted = trial_df.sort_values("time_rel")
                sig = trial_df_sorted[SIGNAL_COL].values
                if len(sig) == len(time_rel):
                    trial_sigs.append(sig)
            if trial_sigs:
                mean_trace = np.nanmean(trial_sigs, axis=0)
                pop_sol_traces.setdefault((pop, sol), []).append(mean_trace)

    return pop_sol_traces, time_rel


def plot_valence_analysis(val_df, pop_sol_traces, time_rel, save_dir):
    """Generate valence analysis figures."""

    # --- Figure 1: Mean traces per solution per population ---
    sol_colors = {
        "water": "#888888",
        "sucrose30": "#2ca02c",
        "nacl150": "#d62728",
        "dry": "#ff7f0e",
    }
    sol_labels = {
        "water": "Water",
        "sucrose30": "Sucrose (30%)",
        "nacl150": "NaCl (150mM)",
        "dry": "Dry",
    }

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.0), sharey=True)
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        ax = axes[pi]
        for sol in ["water", "sucrose30", "nacl150", "dry"]:
            traces = pop_sol_traces.get((pop, sol), [])
            if not traces:
                continue
            arr = np.array(traces)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(time_rel, mean, color=sol_colors[sol], linewidth=1.2,
                    label=sol_labels[sol])
            ax.fill_between(time_rel, mean - sem, mean + sem,
                            color=sol_colors[sol], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.axhline(0, color="gray", linestyle="-", linewidth=0.3)
        ax.set_title(f"{pop} (n={len(pop_sol_traces.get((pop, 'water'), []))})",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("Time from access (s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("z-scored signal")
            ax.legend(fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("Peri-event responses by solution", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "valence_traces_by_solution.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Appetitive vs aversive bar + valence index ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))

    # Panel A: Mean response by category
    ax = axes[0]
    cat_colors = {"Appetitive": "#2ca02c", "Aversive": "#d62728", "Neutral": "#888888"}
    x_pos = np.arange(3)
    width = 0.25
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        pdf = val_df[val_df["population"] == pop]
        means = [pdf["resp_appetitive"].mean(), pdf["resp_aversive"].mean(),
                 pdf["resp_neutral"].mean()]
        sems = [pdf["resp_appetitive"].sem(), pdf["resp_aversive"].sem(),
                pdf["resp_neutral"].sem()]
        offset = (pi - 1) * width
        bars = ax.bar(x_pos + offset, means, width, yerr=sems,
                      color=POP_COLORS[pop], edgecolor="black", linewidth=0.5,
                      capsize=2, label=pop, alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Appetitive", "Aversive", "Neutral"], fontsize=8)
    ax.set_ylabel("Baseline-subtracted\nresponse (z)")
    ax.set_title("Response by valence category", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", linewidth=0.3)

    # Panel B: Valence index by population (swarm-like)
    ax = axes[1]
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        pdf = val_df[val_df["population"] == pop]
        vals = pdf["valence_index"].values
        jitter = np.random.default_rng(42).normal(0, 0.08, len(vals))
        ax.scatter(np.full_like(vals, pi) + jitter, vals,
                   color=POP_COLORS[pop], s=25, alpha=0.7, edgecolors="black",
                   linewidth=0.3)
        ax.errorbar(pi, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                    fmt="o", color="black", markersize=6, capsize=4, linewidth=1.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Cbln4", "Crhbp", "Pnoc"])
    ax.set_ylabel("Valence index\n(app - avers) / (|app| + |avers|)")
    ax.set_title("Valence selectivity", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel C: Appetitive vs aversive selectivity scatter
    ax = axes[2]
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        pdf = val_df[val_df["population"] == pop]
        ax.scatter(pdf["app_selectivity"], pdf["avers_selectivity"],
                   color=POP_COLORS[pop], s=30, alpha=0.7, edgecolors="black",
                   linewidth=0.3, label=pop)
    lim = max(abs(val_df["app_selectivity"]).max(),
              abs(val_df["avers_selectivity"]).max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlabel("Appetitive selectivity\n(sucrose - water)")
    ax.set_ylabel("Aversive selectivity\n(aversive - water)")
    ax.set_title("Selectivity space", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_dir / "valence_encoding_summary.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Individual solution responses per population ---
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=True)
    sols = ["water", "sucrose30", "nacl150", "dry"]
    sol_short = ["Water", "Sucrose", "NaCl", "Dry"]
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        ax = axes[pi]
        pdf = val_df[val_df["population"] == pop]
        sol_vals = [pdf[f"resp_{s}" if s != "sucrose30" else "resp_sucrose"].values
                    if s != "nacl150" else pdf["resp_nacl"].values
                    for s in sols]
        # Fix column mapping
        sol_vals = [
            pdf["resp_water"].values,
            pdf["resp_sucrose"].values,
            pdf["resp_nacl"].values,
            pdf["resp_dry"].values,
        ]
        x = np.arange(4)
        means = [v.mean() for v in sol_vals]
        sems = [v.std() / np.sqrt(len(v)) for v in sol_vals]
        bars_colors = [sol_colors[s] for s in sols]
        ax.bar(x, means, yerr=sems, color=bars_colors, edgecolor="black",
               linewidth=0.5, capsize=3, width=0.6)
        # Overlay individual points
        for xi, vals in enumerate(sol_vals):
            jitter = np.random.default_rng(42+xi).normal(0, 0.06, len(vals))
            ax.scatter(np.full_like(vals, xi) + jitter, vals,
                       color="black", s=10, alpha=0.4, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(sol_short, rotation=45, ha="right", fontsize=8)
        ax.set_title(f"{pop} (n={len(pdf)})", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.axhline(0, color="gray", linewidth=0.3)
        if pi == 0:
            ax.set_ylabel("Baseline-subtracted\nresponse (z)")

    fig.suptitle("Response magnitude by solution", fontsize=11,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "valence_individual_solutions.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# ANALYSIS 2: ENCODING ONSET LATENCY
# ====================================================================

def extract_latencies(results_list):
    """
    From time-resolved GLM results, extract onset latency of significant
    encoding for each predictor per subject.

    Onset = center of the first window where encoding is significant
    (p < alpha). If never significant, latency = NaN.
    """
    rows = []
    for r in results_list:
        subj = r["subject"]
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        tr = r["timeresolved"]
        win_centers = tr["win_centers"]
        chunk_names = tr["chunk_names"]

        for ki, name in enumerate(chunk_names):
            if name == "session":
                continue
            sig_mask = tr["significant"][:, ki]
            if sig_mask.any():
                onset_idx = np.argmax(sig_mask)
                onset_time = win_centers[onset_idx]
                # Also get peak delta-R2 time
                peak_idx = np.argmax(tr["delta_r2"][:, ki])
                peak_time = win_centers[peak_idx]
                peak_dr2 = tr["delta_r2"][peak_idx, ki]
            else:
                onset_time = np.nan
                peak_time = np.nan
                peak_dr2 = 0.0

            rows.append({
                "subject": subj,
                "population": pop,
                "predictor": name,
                "onset_latency": onset_time,
                "peak_latency": peak_time,
                "peak_delta_r2": peak_dr2,
                "n_sig_windows": sig_mask.sum(),
                "total_windows": len(sig_mask),
            })

    return pd.DataFrame(rows)


def plot_latency_analysis(lat_df, results_list, save_dir):
    """Generate timing analysis figures."""

    # --- Figure 1: Onset latency comparison across populations ---
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    predictors = ["solution", "lick_count", "lick_x_solution", "trial_number"]
    pred_labels = ["Solution", "Lick count", "Lick x solution", "Trial number"]

    for ki, (pred, label) in enumerate(zip(predictors, pred_labels)):
        ax = axes[ki]
        pdf = lat_df[lat_df["predictor"] == pred]
        for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
            vals = pdf.loc[pdf["population"] == pop, "onset_latency"].dropna().values
            if len(vals) == 0:
                continue
            jitter = np.random.default_rng(42+pi).normal(0, 0.08, len(vals))
            ax.scatter(np.full_like(vals, pi) + jitter, vals,
                       color=POP_COLORS[pop], s=25, alpha=0.7,
                       edgecolors="black", linewidth=0.3)
            ax.errorbar(pi, np.nanmean(vals),
                        yerr=np.nanstd(vals)/np.sqrt(len(vals)),
                        fmt="o", color="black", markersize=6, capsize=4,
                        linewidth=1.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Cbln4", "Crhbp", "Pnoc"], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ki == 0:
            ax.set_ylabel("Onset latency (s)")

    fig.suptitle("Encoding onset latency by population",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "timing_onset_latency.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Time-resolved delta-R2 overlay (Crhbp vs Pnoc focus) ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    # Compute population-mean time-resolved delta-R2 from individual results
    pop_tr = {}  # (pop, pred_idx) -> list of (n_windows,) arrays
    for r in results_list:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        tr = r["timeresolved"]
        for ki in range(min(4, len(tr["chunk_names"]))):
            pop_tr.setdefault((pop, ki), []).append(tr["delta_r2"][:, ki])

    win_centers = results_list[0]["timeresolved"]["win_centers"]
    chunk_names = results_list[0]["timeresolved"]["chunk_names"]

    for ki, (pred, label) in enumerate(zip(predictors, pred_labels)):
        ax = axes[ki // 2, ki % 2]
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            traces = pop_tr.get((pop, ki), [])
            if not traces:
                continue
            arr = np.array(traces)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(win_centers, mean, color=POP_COLORS[pop], linewidth=1.5,
                    label=pop)
            ax.fill_between(win_centers, mean - sem, mean + sem,
                            color=POP_COLORS[pop], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("$\\Delta R^2$")
        if ki >= 2:
            ax.set_xlabel("Time from access (s)")
        if ki == 0:
            ax.legend(fontsize=7, frameon=False)

    fig.suptitle("Time-resolved encoding: population comparison",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / "timing_timeresolved_overlay.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Crhbp vs Pnoc direct comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    # Panel A: Onset latency paired comparison for solution encoding
    ax = axes[0]
    for pred, marker, offset in [("solution", "o", -0.15), ("lick_count", "s", 0.15)]:
        pdf = lat_df[lat_df["predictor"] == pred]
        for pi, pop in enumerate(["Crhbp", "Pnoc"]):
            vals = pdf.loc[pdf["population"] == pop, "onset_latency"].dropna().values
            jitter = np.random.default_rng(42+pi).normal(0, 0.05, len(vals))
            ax.scatter(np.full_like(vals, pi) + jitter + offset, vals,
                       color=POP_COLORS[pop], s=30, alpha=0.7,
                       edgecolors="black", linewidth=0.3, marker=marker,
                       label=f"{pop} ({pred})" if pi == 0 or True else "")
            ax.errorbar(pi + offset, np.nanmean(vals),
                        yerr=np.nanstd(vals)/np.sqrt(len(vals)),
                        fmt=marker, color="black", markersize=7, capsize=4,
                        linewidth=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Crhbp", "Pnoc"])
    ax.set_ylabel("Onset latency (s)")
    ax.set_title("Crhbp vs Pnoc: onset latency", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: Peak latency comparison
    ax = axes[1]
    for pred, marker, offset in [("solution", "o", -0.15), ("lick_count", "s", 0.15)]:
        pdf = lat_df[lat_df["predictor"] == pred]
        for pi, pop in enumerate(["Crhbp", "Pnoc"]):
            vals = pdf.loc[pdf["population"] == pop, "peak_latency"].dropna().values
            jitter = np.random.default_rng(42+pi).normal(0, 0.05, len(vals))
            ax.scatter(np.full_like(vals, pi) + jitter + offset, vals,
                       color=POP_COLORS[pop], s=30, alpha=0.7,
                       edgecolors="black", linewidth=0.3, marker=marker,
                       label=f"{pop} ({pred})" if pi == 0 or True else "")
            ax.errorbar(pi + offset, np.nanmean(vals),
                        yerr=np.nanstd(vals)/np.sqrt(len(vals)),
                        fmt=marker, color="black", markersize=7, capsize=4,
                        linewidth=1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Crhbp", "Pnoc"])
    ax.set_ylabel("Peak latency (s)")
    ax.set_title("Crhbp vs Pnoc: peak latency", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, frameon=False, ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_dir / "timing_crhbp_vs_pnoc.png",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ====================================================================
# STATISTICS
# ====================================================================

def run_valence_stats(val_df):
    """Run statistical tests on valence metrics."""
    print("\n" + "="*70)
    print("  VALENCE ENCODING STATISTICS")
    print("="*70)

    # Test if valence index differs from zero per population
    print("\nValence index vs. zero (one-sample Wilcoxon):")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        vals = val_df.loc[val_df["population"] == pop, "valence_index"].values
        if len(vals) > 2:
            stat, p = wilcoxon(vals)
            direction = "appetitive" if np.median(vals) > 0 else "aversive"
            print(f"  {pop}: median={np.median(vals):.3f}, "
                  f"W={stat:.1f}, p={p:.4f} ({direction}-biased)")

    # Group comparison of valence index
    groups = [val_df.loc[val_df["population"] == p, "valence_index"].values
              for p in ["Cbln4", "Crhbp", "Pnoc"]]
    H, p = kruskal(*groups)
    print(f"\nValence index Kruskal-Wallis: H={H:.2f}, p={p:.4f}")
    if p < 0.05:
        for i, j in [(0,1), (0,2), (1,2)]:
            pops = ["Cbln4", "Crhbp", "Pnoc"]
            U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            print(f"  {pops[i]} vs {pops[j]}: U={U:.0f}, p={pw:.4f}")

    # Appetitive and aversive selectivity vs zero
    print("\nAppetitive selectivity vs. zero:")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        vals = val_df.loc[val_df["population"] == pop, "app_selectivity"].values
        if len(vals) > 2:
            stat, p = wilcoxon(vals)
            print(f"  {pop}: mean={np.mean(vals):.4f}, W={stat:.1f}, p={p:.4f}")

    print("\nAversive selectivity vs. zero:")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        vals = val_df.loc[val_df["population"] == pop, "avers_selectivity"].values
        if len(vals) > 2:
            stat, p = wilcoxon(vals)
            print(f"  {pop}: mean={np.mean(vals):.4f}, W={stat:.1f}, p={p:.4f}")

    # Group comparisons for appetitive and aversive selectivity
    for metric, label in [("app_selectivity", "Appetitive"), ("avers_selectivity", "Aversive")]:
        groups = [val_df.loc[val_df["population"] == p, metric].values
                  for p in ["Cbln4", "Crhbp", "Pnoc"]]
        H, p_kw = kruskal(*groups)
        print(f"\n{label} selectivity Kruskal-Wallis: H={H:.2f}, p={p_kw:.4f}")
        if p_kw < 0.05:
            for i, j in [(0,1), (0,2), (1,2)]:
                pops = ["Cbln4", "Crhbp", "Pnoc"]
                U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                print(f"  {pops[i]} vs {pops[j]}: U={U:.0f}, p={pw:.4f}")

    # Per-solution group comparisons
    print("\nPer-solution response Kruskal-Wallis:")
    for sol_col, sol_name in [("resp_water", "Water"), ("resp_sucrose", "Sucrose"),
                               ("resp_nacl", "NaCl"), ("resp_dry", "Dry")]:
        groups = [val_df.loc[val_df["population"] == p, sol_col].values
                  for p in ["Cbln4", "Crhbp", "Pnoc"]]
        H, p_kw = kruskal(*groups)
        sig = "***" if p_kw < 0.001 else ("**" if p_kw < 0.01 else ("*" if p_kw < 0.05 else "ns"))
        print(f"  {sol_name}: H={H:.2f}, p={p_kw:.4f} {sig}")
        if p_kw < 0.05:
            for i, j in [(0,1), (0,2), (1,2)]:
                pops = ["Cbln4", "Crhbp", "Pnoc"]
                U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                pw_str = "*" if pw < 0.05 else "ns"
                print(f"    {pops[i]} vs {pops[j]}: U={U:.0f}, p={pw:.4f} {pw_str}")


def run_latency_stats(lat_df):
    """Run statistical tests on encoding latencies."""
    print("\n" + "="*70)
    print("  ENCODING LATENCY STATISTICS")
    print("="*70)

    predictors = ["solution", "lick_count", "lick_x_solution"]
    pred_labels = ["Solution", "Lick count", "Lick x solution"]

    for pred, label in zip(predictors, pred_labels):
        pdf = lat_df[lat_df["predictor"] == pred]
        print(f"\n--- {label} encoding ---")

        # Summary per population
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"] == pop, "onset_latency"]
            n_sig = vals.notna().sum()
            n_total = len(vals)
            print(f"  {pop}: {n_sig}/{n_total} subjects with significant encoding")
            if n_sig > 0:
                v = vals.dropna().values
                print(f"    onset: {np.mean(v):.2f} +/- {np.std(v):.2f} s "
                      f"(median={np.median(v):.2f})")
                pk = pdf.loc[(pdf["population"] == pop) & pdf["peak_latency"].notna(),
                             "peak_latency"].values
                if len(pk) > 0:
                    print(f"    peak:  {np.mean(pk):.2f} +/- {np.std(pk):.2f} s "
                          f"(median={np.median(pk):.2f})")

        # Kruskal-Wallis on onset latency (only among significant subjects)
        groups = []
        pop_names = []
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"] == pop, "onset_latency"].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
                pop_names.append(pop)

        if len(groups) >= 2:
            if len(groups) == 3:
                H, p = kruskal(*groups)
                print(f"\n  Onset latency KW: H={H:.2f}, p={p:.4f}")
            # Pairwise Crhbp vs Pnoc specifically
            for i in range(len(pop_names)):
                for j in range(i+1, len(pop_names)):
                    U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                    print(f"  {pop_names[i]} vs {pop_names[j]} onset: "
                          f"U={U:.0f}, p={pw:.4f}")

        # Same for peak latency
        groups_pk = []
        pop_names_pk = []
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"] == pop, "peak_latency"].dropna().values
            if len(vals) >= 2:
                groups_pk.append(vals)
                pop_names_pk.append(pop)

        if len(groups_pk) >= 2:
            for i in range(len(pop_names_pk)):
                for j in range(i+1, len(pop_names_pk)):
                    U, pw = mannwhitneyu(groups_pk[i], groups_pk[j],
                                         alternative='two-sided')
                    print(f"  {pop_names_pk[i]} vs {pop_names_pk[j]} peak: "
                          f"U={U:.0f}, p={pw:.4f}")


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_h5_data(H5)
    print(f"Loaded {len(df)} rows, {df['subject'].nunique()} subjects")

    # Analysis 1: Valence encoding
    print("\n--- Computing valence metrics ---")
    val_df = compute_valence_metrics(df)
    print(f"Valence metrics for {len(val_df)} subjects")

    print("\n--- Computing population traces ---")
    pop_sol_traces, time_rel = compute_population_traces(df)

    print("\n--- Plotting valence analysis ---")
    plot_valence_analysis(val_df, pop_sol_traces, time_rel, SAVE_DIR)

    # Run valence stats
    run_valence_stats(val_df)

    # Analysis 2: Encoding latency
    print("\n\n--- Loading time-resolved GLM results ---")
    b1 = pickle.load(open(SAVE_DIR / "_batch1_results.pkl", "rb"))
    b2 = pickle.load(open(SAVE_DIR / "_batch2_results.pkl", "rb"))
    all_results = b1 + b2

    print("--- Extracting encoding latencies ---")
    lat_df = extract_latencies(all_results)
    print(f"Latency data for {lat_df['subject'].nunique()} subjects")

    print("\n--- Plotting timing analysis ---")
    plot_latency_analysis(lat_df, all_results, SAVE_DIR)

    # Run latency stats
    run_latency_stats(lat_df)

    # Save tables
    val_df.to_csv(SAVE_DIR / "valence_metrics.csv", index=False)
    lat_df.to_csv(SAVE_DIR / "encoding_latencies.csv", index=False)
    print(f"\nSaved valence_metrics.csv and encoding_latencies.csv")
    print("Done.")
