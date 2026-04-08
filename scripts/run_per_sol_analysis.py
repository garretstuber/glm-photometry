"""
Group stats, timing, and figures for per-solution GLM.
"""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu

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
SOL_COLORS = {"sucrose30": "#2ca02c", "nacl150": "#d62728", "dry": "#ff7f0e",
              "lick_count": "#9467bd", "trial_number": "#8c564b"}
SOL_LABELS = {"sucrose30": "Sucrose", "nacl150": "NaCl", "dry": "Dry",
              "lick_count": "Lick count", "trial_number": "Trial num."}

def load_results(prefix):
    b1 = pickle.load(open(SAVE_DIR / f"{prefix}batch1_results.pkl", "rb"))
    b2 = pickle.load(open(SAVE_DIR / f"{prefix}batch2_results.pkl", "rb"))
    return b1 + b2

results = load_results("_per_sol_")
chunk_names = results[0]["chunk_names"]
task_idx = [i for i, n in enumerate(chunk_names) if n != "session"]
task_names = [chunk_names[i] for i in task_idx]

pops = {}
for r in results:
    pop = POP_MAP.get(r["channel_id"], r["channel_id"])
    pops.setdefault(pop, []).append(r)

# ── Summary stats ──
print("="*70)
print("  PER-SOLUTION GLM RESULTS")
print("="*70)
for pop in ["Cbln4", "Crhbp", "Pnoc"]:
    rr = pops[pop]
    n = len(rr)
    r2s = [r["full_glm"]["r2_full"] for r in rr]
    print(f"\n{pop} (n={n}): R2 = {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}")
    for ki in task_idx:
        name = chunk_names[ki]
        dr2s = [r["full_glm"]["delta_r2_obs"][ki] for r in rr]
        sigs = [r["full_glm"]["significant"][ki] for r in rr]
        print(f"  {SOL_LABELS.get(name,name):15s}  dR2={np.mean(dr2s):.4f}+/-{np.std(dr2s):.4f}  "
              f"sig: {sum(sigs)}/{n} ({100*np.mean(sigs):.0f}%)")

# Group-level KW
print(f"\nGroup-level Kruskal-Wallis:")
for ki in task_idx:
    name = chunk_names[ki]
    groups = [[r["full_glm"]["delta_r2_obs"][ki] for r in pops[p]] for p in ["Cbln4","Crhbp","Pnoc"]]
    H, p = kruskal(*groups)
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
    print(f"  {SOL_LABELS.get(name,name):15s}  H={H:.2f}  p={p:.4f}  {sig}")
    if p < 0.05:
        for i, j in [(0,1),(0,2),(1,2)]:
            pn = ["Cbln4","Crhbp","Pnoc"]
            U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
            pw_s = "*" if pw < 0.05 else "ns"
            print(f"    {pn[i]} vs {pn[j]}: U={U:.0f}, p={pw:.4f} {pw_s}")

# ── Figure 1: DR2 bar per population ──
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
    ax = axes[pi]
    rr = pops[pop]
    x = np.arange(len(task_idx))
    dr2_means = [np.mean([r["full_glm"]["delta_r2_obs"][i] for r in rr]) for i in task_idx]
    dr2_sems = [np.std([r["full_glm"]["delta_r2_obs"][i] for r in rr])/np.sqrt(len(rr)) for i in task_idx]
    sig_pcts = [np.mean([r["full_glm"]["significant"][i] for r in rr]) for i in task_idx]
    cols = [SOL_COLORS.get(task_names[k], "#999") for k in range(len(task_idx))]
    ax.bar(x, dr2_means, yerr=dr2_sems, color=cols, edgecolor="black",
           linewidth=0.5, capsize=3, width=0.6)
    for xi in range(len(task_idx)):
        ax.text(xi, dr2_means[xi] + dr2_sems[xi] + 0.001,
                f"{sig_pcts[xi]*100:.0f}%", ha="center", fontsize=7, fontweight="bold")
    labels = [SOL_LABELS.get(task_names[k], task_names[k]) for k in range(len(task_idx))]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_title(f"{pop} (n={len(rr)})", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if pi == 0:
        ax.set_ylabel("$\\Delta R^2$")
fig.suptitle("Per-solution GLM: unique variance explained (% sig above bars)",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(SAVE_DIR / "glm_dr2_bar_per_sol.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Figure 2: Heatmap of % significant ──
fig, ax = plt.subplots(figsize=(6, 3.5))
pop_order = ["Cbln4", "Crhbp", "Pnoc"]
mat = np.zeros((3, len(task_idx)))
for pi, pop in enumerate(pop_order):
    rr = pops[pop]
    for ki_local, ki in enumerate(task_idx):
        mat[pi, ki_local] = 100 * np.mean([r["full_glm"]["significant"][ki] for r in rr])

im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=100)
ax.set_xticks(range(len(task_idx)))
ax.set_xticklabels([SOL_LABELS.get(task_names[k], task_names[k]) for k in range(len(task_idx))],
                    rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(3))
ax.set_yticklabels(pop_order, fontsize=10)
for pi in range(3):
    for ki in range(len(task_idx)):
        ax.text(ki, pi, f"{mat[pi,ki]:.0f}%", ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if mat[pi,ki] > 60 else "black")
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("% subjects significant", fontsize=9)
ax.set_title("Encoding prevalence by solution and population", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(SAVE_DIR / "glm_heatmap_per_sol.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Figure 3: Time-resolved overlay per solution ──
sol_preds = ["sucrose30", "nacl150", "dry"]
fig, axes = plt.subplots(1, 3, figsize=(11, 3.0), sharey=True)
win_centers = results[0]["timeresolved"]["win_centers"]
for si, sol in enumerate(sol_preds):
    ax = axes[si]
    sol_ki = chunk_names.index(sol)
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        traces = [r["timeresolved"]["delta_r2"][:, sol_ki] for r in pops[pop]]
        arr = np.array(traces)
        mean = np.nanmean(arr, axis=0)
        sem = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
        ax.plot(win_centers, mean, color=POP_COLORS[pop], linewidth=1.5, label=pop)
        ax.fill_between(win_centers, mean-sem, mean+sem, color=POP_COLORS[pop], alpha=0.2)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.set_title(f"{SOL_LABELS[sol]} vs Water", fontsize=10, fontweight="bold")
    ax.set_xlabel("Time from access (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if si == 0:
        ax.set_ylabel("$\\Delta R^2$")
        ax.legend(fontsize=7, frameon=False)
fig.suptitle("Time-resolved encoding by individual solution contrast",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(SAVE_DIR / "timing_timeresolved_per_sol.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ── Extract latencies ──
def extract_latencies(results_list):
    rows = []
    for r in results_list:
        subj = r["subject"]
        pop = POP_MAP.get(r["channel_id"], r["channel_id"])
        tr = r["timeresolved"]
        wc = tr["win_centers"]
        for ki, name in enumerate(tr["chunk_names"]):
            if name == "session":
                continue
            sig_mask = tr["significant"][:, ki]
            if sig_mask.any():
                onset_time = wc[np.argmax(sig_mask)]
                peak_idx = np.argmax(tr["delta_r2"][:, ki])
                peak_time = wc[peak_idx]
            else:
                onset_time = np.nan
                peak_time = np.nan
            rows.append({"subject": subj, "population": pop, "predictor": name,
                         "onset_latency": onset_time, "peak_latency": peak_time})
    return pd.DataFrame(rows)

lat_df = extract_latencies(results)

# Latency stats for solution contrasts
print(f"\n\n{'='*70}")
print("  ENCODING ONSET LATENCY: PER-SOLUTION CONTRASTS")
print("="*70)
for pred in ["sucrose30", "nacl150", "dry", "lick_count"]:
    pdf = lat_df[lat_df["predictor"] == pred]
    label = SOL_LABELS.get(pred, pred)
    print(f"\n--- {label} ---")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        vals = pdf.loc[pdf["population"]==pop, "onset_latency"]
        n_sig = vals.notna().sum()
        n_total = len(vals)
        print(f"  {pop}: {n_sig}/{n_total} sig", end="")
        if n_sig > 0:
            v = vals.dropna().values
            print(f"  onset={np.mean(v):.2f}+/-{np.std(v):.2f} (med={np.median(v):.2f})")
        else:
            print()
    groups, pnames = [], []
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        v = pdf.loc[pdf["population"]==pop, "onset_latency"].dropna().values
        if len(v) >= 2:
            groups.append(v)
            pnames.append(pop)
    if len(groups) == 3:
        H, p = kruskal(*groups)
        print(f"  KW onset: H={H:.2f}, p={p:.4f}")
    if len(groups) >= 2:
        for i in range(len(pnames)):
            for j in range(i+1, len(pnames)):
                U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                print(f"  {pnames[i]} vs {pnames[j]}: U={U:.0f}, p={pw:.4f}")

# ── Figure 4: Onset latency per solution ──
fig, axes = plt.subplots(1, 3, figsize=(9, 3.5))
for si, sol in enumerate(["sucrose30", "nacl150", "dry"]):
    ax = axes[si]
    pdf = lat_df[lat_df["predictor"] == sol]
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        vals = pdf.loc[pdf["population"]==pop, "onset_latency"].dropna().values
        if len(vals) == 0:
            continue
        jitter = np.random.default_rng(42+pi).normal(0, 0.08, len(vals))
        ax.scatter(np.full_like(vals, pi) + jitter, vals,
                   color=POP_COLORS[pop], s=25, alpha=0.7, edgecolors="black", linewidth=0.3)
        ax.errorbar(pi, np.nanmean(vals), yerr=np.nanstd(vals)/np.sqrt(len(vals)),
                    fmt="o", color="black", markersize=6, capsize=4, linewidth=1.5)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["Cbln4", "Crhbp", "Pnoc"], fontsize=8)
    ax.set_title(f"{SOL_LABELS[sol]} vs Water", fontsize=10, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if si == 0:
        ax.set_ylabel("Onset latency (s)")
fig.suptitle("Encoding onset latency by solution contrast",
             fontsize=11, fontweight="bold", y=1.02)
fig.tight_layout()
fig.savefig(SAVE_DIR / "timing_onset_per_sol.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# Save
lat_df.to_csv(SAVE_DIR / "encoding_latencies_per_sol.csv", index=False)

# Save per-subject results CSV
rows = []
for r in results:
    pop = POP_MAP.get(r["channel_id"], r["channel_id"])
    for ki in task_idx:
        rows.append({
            "subject": r["subject"], "population": pop,
            "predictor": chunk_names[ki],
            "delta_r2": r["full_glm"]["delta_r2_obs"][ki],
            "p_value": r["full_glm"]["pvals"][ki],
            "significant": r["full_glm"]["significant"][ki],
            "r2_full": r["full_glm"]["r2_full"],
        })
pd.DataFrame(rows).to_csv(SAVE_DIR / "glm_per_solution_results.csv", index=False)
print("\nSaved CSVs and figures. Done.")
