"""
Run group statistics, timing extraction, and figure generation for the
reduced GLM (no lick x solution interaction). Valence metrics are reused
from the previous analysis since they are GLM-independent.

Produces:
  - Figures with _no_lxs suffix
  - Comparison of full vs reduced model
  - Encoding latency extraction
"""
import pickle, sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, wilcoxon

sys.path.insert(0, str(Path("mnt/20260331 - Elerding et al., VTA GABA neurons")))
from fp_glm_group_analysis import load_h5_data

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    matplotlib.font_manager.fontManager.addfont(
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf")
    plt.rcParams["font.family"] = "Liberation Sans"
except Exception:
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8,
                      "xtick.major.width": 0.8, "ytick.major.width": 0.8})

SAVE_DIR = Path("mnt/20260331 - Elerding et al., VTA GABA neurons")
POP_MAP = {"cbln4": "Cbln4", "crhbp": "Crhbp", "pnoc": "Pnoc"}
POP_COLORS = {"Cbln4": "#4C72B0", "Crhbp": "#DD8452", "Pnoc": "#55A868"}


def load_results(prefix):
    b1 = pickle.load(open(SAVE_DIR / f"{prefix}batch1_results.pkl", "rb"))
    b2 = pickle.load(open(SAVE_DIR / f"{prefix}batch2_results.pkl", "rb"))
    return b1 + b2


def summarize_glm(results, label):
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    pops = {}
    for r in results:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        pops.setdefault(pop, []).append(r)

    chunk_names = results[0]["chunk_names"]
    task_idx = [i for i, n in enumerate(chunk_names) if n != "session"]

    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        rr = pops[pop]
        n = len(rr)
        r2s = [r["full_glm"]["r2_full"] for r in rr]
        print(f"\n{pop} (n={n}):")
        print(f"  Full model R2: {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
        for ki in task_idx:
            name = chunk_names[ki]
            dr2s = [r["full_glm"]["delta_r2_obs"][ki] for r in rr]
            sigs = [r["full_glm"]["significant"][ki] for r in rr]
            pct_sig = 100 * np.mean(sigs)
            print(f"  {name:20s}  dR2={np.mean(dr2s):.4f} +/- {np.std(dr2s):.4f}  "
                  f"sig: {sum(sigs)}/{n} ({pct_sig:.0f}%)")

    print(f"\n  Group-level Kruskal-Wallis tests:")
    for ki in task_idx:
        name = chunk_names[ki]
        groups = [[r["full_glm"]["delta_r2_obs"][ki] for r in pops[pop]]
                  for pop in ["Cbln4", "Crhbp", "Pnoc"]]
        H, p = kruskal(*groups)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
        print(f"  {name:20s}  H={H:.2f}  p={p:.4f}  {sig}")
        if p < 0.05:
            for i, j in [(0,1), (0,2), (1,2)]:
                pn = ["Cbln4", "Crhbp", "Pnoc"]
                U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                pw_str = "*" if pw < 0.05 else "ns"
                print(f"    {pn[i]} vs {pn[j]}: U={U:.0f}, p={pw:.4f} {pw_str}")
    return pops


def extract_latencies(results_list):
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
                peak_idx = np.argmax(tr["delta_r2"][:, ki])
                peak_time = win_centers[peak_idx]
                peak_dr2 = tr["delta_r2"][peak_idx, ki]
            else:
                onset_time = np.nan
                peak_time = np.nan
                peak_dr2 = 0.0
            rows.append({
                "subject": subj, "population": pop, "predictor": name,
                "onset_latency": onset_time, "peak_latency": peak_time,
                "peak_delta_r2": peak_dr2,
                "n_sig_windows": sig_mask.sum(),
                "total_windows": len(sig_mask),
            })
    return pd.DataFrame(rows)


def plot_timing(lat_df, results_list, save_dir, suffix="_no_lxs"):
    predictors = ["solution", "lick_count", "trial_number"]
    pred_labels = ["Solution", "Lick count", "Trial number"]

    # Onset latency figure
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
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
                        fmt="o", color="black", markersize=6, capsize=4, linewidth=1.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Cbln4", "Crhbp", "Pnoc"], fontsize=8)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ki == 0:
            ax.set_ylabel("Onset latency (s)")
    fig.suptitle("Encoding onset latency (reduced GLM, no lick x sol.)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / f"timing_onset_latency{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Time-resolved overlay
    pop_tr = {}
    for r in results_list:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        tr = r["timeresolved"]
        for ki in range(min(3, len(tr["chunk_names"]))):
            if tr["chunk_names"][ki] == "session":
                continue
            pop_tr.setdefault((pop, tr["chunk_names"][ki]), []).append(tr["delta_r2"][:, ki])

    win_centers = results_list[0]["timeresolved"]["win_centers"]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.0))
    for ki, (pred, label) in enumerate(zip(predictors, pred_labels)):
        ax = axes[ki]
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            traces = pop_tr.get((pop, pred), [])
            if not traces:
                continue
            arr = np.array(traces)
            mean = np.nanmean(arr, axis=0)
            sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            ax.plot(win_centers, mean, color=POP_COLORS[pop], linewidth=1.5, label=pop)
            ax.fill_between(win_centers, mean - sem, mean + sem,
                            color=POP_COLORS[pop], alpha=0.2)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("$\\Delta R^2$")
        ax.set_xlabel("Time from access (s)")
        if ki == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Time-resolved encoding (reduced GLM, no lick x sol.)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / f"timing_timeresolved_overlay{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Crhbp vs Pnoc direct
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))
    for pred, marker, offset in [("solution", "o", -0.15), ("lick_count", "s", 0.15)]:
        pdf = lat_df[lat_df["predictor"] == pred]
        for panel, metric, ylabel in [(0, "onset_latency", "Onset latency (s)"),
                                       (1, "peak_latency", "Peak latency (s)")]:
            ax = axes[panel]
            for pi, pop in enumerate(["Crhbp", "Pnoc"]):
                vals = pdf.loc[pdf["population"] == pop, metric].dropna().values
                jitter = np.random.default_rng(42+pi).normal(0, 0.05, len(vals))
                ax.scatter(np.full_like(vals, pi) + jitter + offset, vals,
                           color=POP_COLORS[pop], s=30, alpha=0.7,
                           edgecolors="black", linewidth=0.3, marker=marker,
                           label=f"{pop} ({pred})" if panel == 0 else "")
                ax.errorbar(pi + offset, np.nanmean(vals),
                            yerr=np.nanstd(vals)/np.sqrt(len(vals)),
                            fmt=marker, color="black", markersize=7, capsize=4, linewidth=1.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Crhbp", "Pnoc"])
            ax.set_ylabel(ylabel)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0].set_title("Crhbp vs Pnoc: onset latency", fontsize=10, fontweight="bold")
    axes[1].set_title("Crhbp vs Pnoc: peak latency", fontsize=10, fontweight="bold")
    axes[0].legend(fontsize=6, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(save_dir / f"timing_crhbp_vs_pnoc{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # DR2 bar comparison
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=True)
    pred_colors = {"solution": "#4C72B0", "lick_count": "#DD8452", "trial_number": "#C44E52"}
    pops_data = {}
    for r in results_list:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        pops_data.setdefault(pop, []).append(r)

    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        ax = axes[pi]
        rr = pops_data[pop]
        chunk_names = rr[0]["chunk_names"]
        task_idx = [i for i, n in enumerate(chunk_names) if n != "session"]
        names = [chunk_names[i] for i in task_idx]
        x = np.arange(len(task_idx))
        dr2_means = [np.mean([r["full_glm"]["delta_r2_obs"][i] for r in rr]) for i in task_idx]
        dr2_sems = [np.std([r["full_glm"]["delta_r2_obs"][i] for r in rr])/np.sqrt(len(rr)) for i in task_idx]
        sig_pcts = [np.mean([r["full_glm"]["significant"][i] for r in rr]) for i in task_idx]
        colors = [pred_colors.get(n, "#999999") for n in names]
        bars = ax.bar(x, dr2_means, yerr=dr2_sems, color=colors, edgecolor="black",
                      linewidth=0.5, capsize=3, width=0.6)
        for xi in range(len(task_idx)):
            pct = sig_pcts[xi] * 100
            ax.text(xi, dr2_means[xi] + dr2_sems[xi] + 0.002, f"{pct:.0f}%",
                    ha="center", fontsize=7, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{pop} (n={len(rr)})", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("$\\Delta R^2$ (mean +/- SEM)")
    fig.suptitle("Reduced GLM: unique variance explained (% sig above bars)",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_dir / f"glm_dr2_bar{suffix}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_latency_stats(lat_df):
    print("\n" + "="*70)
    print("  ENCODING LATENCY STATISTICS (reduced GLM)")
    print("="*70)
    predictors = ["solution", "lick_count"]
    for pred in predictors:
        pdf = lat_df[lat_df["predictor"] == pred]
        label = pred.replace("_", " ").title()
        print(f"\n--- {label} encoding ---")
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"] == pop, "onset_latency"]
            n_sig = vals.notna().sum()
            n_total = len(vals)
            print(f"  {pop}: {n_sig}/{n_total} sig")
            if n_sig > 0:
                v = vals.dropna().values
                print(f"    onset: {np.mean(v):.2f} +/- {np.std(v):.2f} s (median={np.median(v):.2f})")
                pk = pdf.loc[(pdf["population"]==pop) & pdf["peak_latency"].notna(), "peak_latency"].values
                if len(pk) > 0:
                    print(f"    peak:  {np.mean(pk):.2f} +/- {np.std(pk):.2f} s (median={np.median(pk):.2f})")

        groups, pnames = [], []
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"]==pop, "onset_latency"].dropna().values
            if len(vals) >= 2:
                groups.append(vals)
                pnames.append(pop)
        if len(groups) >= 2:
            if len(groups) == 3:
                H, p = kruskal(*groups)
                print(f"  Onset KW: H={H:.2f}, p={p:.4f}")
            for i in range(len(pnames)):
                for j in range(i+1, len(pnames)):
                    U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                    print(f"  {pnames[i]} vs {pnames[j]} onset: U={U:.0f}, p={pw:.4f}")

        groups_pk, pnames_pk = [], []
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            vals = pdf.loc[pdf["population"]==pop, "peak_latency"].dropna().values
            if len(vals) >= 2:
                groups_pk.append(vals)
                pnames_pk.append(pop)
        if "Crhbp" in pnames_pk and "Pnoc" in pnames_pk:
            ci, pi_ = pnames_pk.index("Crhbp"), pnames_pk.index("Pnoc")
            U, pw = mannwhitneyu(groups_pk[ci], groups_pk[pi_], alternative='two-sided')
            print(f"  Crhbp vs Pnoc peak: U={U:.0f}, p={pw:.4f}")


# ── MAIN ──
if __name__ == "__main__":
    print("Loading reduced GLM results...")
    no_lxs = load_results("_no_lxs_")
    print(f"Loaded {len(no_lxs)} subjects")

    # Also load original for comparison
    print("Loading original (full) GLM results...")
    full = load_results("_")

    # Summarize both
    summarize_glm(full, "FULL MODEL (with lick x solution)")
    pops_no_lxs = summarize_glm(no_lxs, "REDUCED MODEL (no lick x solution)")

    # Side-by-side comparison
    print(f"\n\n{'='*70}")
    print("  SIDE-BY-SIDE: Full vs Reduced model")
    print(f"{'='*70}")
    full_pops = {}
    for r in full:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        full_pops.setdefault(pop, []).append(r)
    no_lxs_pops = {}
    for r in no_lxs:
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        no_lxs_pops.setdefault(pop, []).append(r)

    print(f"\n  Mean full-model R2:")
    print(f"  {'Pop':<10} {'Full model':>12} {'Reduced':>12}")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        f = np.mean([r["full_glm"]["r2_full"] for r in full_pops[pop]])
        r = np.mean([r["full_glm"]["r2_full"] for r in no_lxs_pops[pop]])
        print(f"  {pop:<10} {f:>12.4f} {r:>12.4f}")

    # Compare shared predictors
    shared = ["solution", "lick_count", "trial_number"]
    for pred in shared:
        print(f"\n  {pred} -- delta-R2 / % significant:")
        print(f"  {'Pop':<10} {'Full dR2':>10} {'Red dR2':>10} {'Full %sig':>10} {'Red %sig':>10}")
        full_cn = full[0]["chunk_names"]
        red_cn = no_lxs[0]["chunk_names"]
        fi = full_cn.index(pred)
        ri = red_cn.index(pred)
        for pop in ["Cbln4", "Crhbp", "Pnoc"]:
            fd = np.mean([r["full_glm"]["delta_r2_obs"][fi] for r in full_pops[pop]])
            rd = np.mean([r["full_glm"]["delta_r2_obs"][ri] for r in no_lxs_pops[pop]])
            fs = 100*np.mean([r["full_glm"]["significant"][fi] for r in full_pops[pop]])
            rs = 100*np.mean([r["full_glm"]["significant"][ri] for r in no_lxs_pops[pop]])
            print(f"  {pop:<10} {fd:>10.4f} {rd:>10.4f} {fs:>9.0f}% {rs:>9.0f}%")

    # Timing analysis
    print("\n\n--- Extracting encoding latencies (reduced GLM) ---")
    lat_df = extract_latencies(no_lxs)
    print(f"Latency data for {lat_df['subject'].nunique()} subjects")

    # Plot timing figures
    print("--- Plotting timing figures ---")
    plot_timing(lat_df, no_lxs, SAVE_DIR, suffix="_no_lxs")

    # Latency stats
    run_latency_stats(lat_df)

    # Save
    lat_df.to_csv(SAVE_DIR / "encoding_latencies_no_lxs.csv", index=False)
    print(f"\nSaved encoding_latencies_no_lxs.csv and figures with _no_lxs suffix")
    print("Done.")
