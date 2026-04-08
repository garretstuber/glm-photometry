"""
Two analyses:
  1. Single-trial decoding: classify solution identity from photometry signal
  2. Early vs. late trial split: compare encoding strength across session halves

Elerding et al. -- VTA GABA neuron subtypes
"""
import sys, pickle, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import kruskal, mannwhitneyu, wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, balanced_accuracy_score

sys.path.insert(0, str(Path("mnt/20260331 - Elerding et al., VTA GABA neurons")))
from fp_glm_group_analysis import load_h5_data
from fp_glm_circular_shift import (
    get_trial_matrix, run_permutation_glm, N_SHIFTS, ALPHA,
)
from run_batch_per_solution import build_per_solution_dm

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
H5 = SAVE_DIR / "streams_peth_multi_mix_all.h5"
SIGNAL_COL = "delta_signal_poly_zscore_blsub"
POP_MAP = {"cbln4": "Cbln4", "crhbp": "Crhbp", "pnoc": "Pnoc"}
POP_COLORS = {"Cbln4": "#4C72B0", "Crhbp": "#DD8452", "Pnoc": "#55A868"}
SOL_ORDER = ["water", "sucrose30", "nacl150", "dry"]
SOL_LABELS = {"water": "Water", "sucrose30": "Sucrose", "nacl150": "NaCl", "dry": "Dry"}

# Response window for decoding features
RESP_WIN = (0.0, 3.0)
BL_WIN = (-2.0, -0.5)
EARLY_WIN = (0.0, 1.5)
LATE_WIN = (1.5, 3.0)


# ====================================================================
# ANALYSIS 1: SINGLE-TRIAL DECODING
# ====================================================================

def extract_trial_features(df, subject):
    """
    Extract per-trial feature vectors for decoding.

    Features per trial:
      - mean response in response window (baseline-subtracted)
      - mean response in early window
      - mean response in late window
      - peak response
      - response slope (late - early)

    Returns X (n_trials, n_features), y (n_trials,) solution labels
    """
    df_subj = df[df["subject"] == subject]
    sessions = sorted(df_subj["blockname"].unique())

    trials_X = []
    trials_y = []

    for sess in sessions:
        df_sess = df_subj[df_subj["blockname"] == sess]
        time_rel = np.sort(df_sess["time_rel"].unique())

        bl_mask = (time_rel >= BL_WIN[0]) & (time_rel < BL_WIN[1])
        resp_mask = (time_rel >= RESP_WIN[0]) & (time_rel < RESP_WIN[1])
        early_mask = (time_rel >= EARLY_WIN[0]) & (time_rel < EARLY_WIN[1])
        late_mask = (time_rel >= LATE_WIN[0]) & (time_rel < LATE_WIN[1])

        trial_groups = df_sess.groupby(["solution", "event_number_solution"])
        for (sol, evt), trial_df in trial_groups:
            trial_sorted = trial_df.sort_values("time_rel")
            sig = trial_sorted[SIGNAL_COL].values
            tr = trial_sorted["time_rel"].values

            if len(sig) != len(time_rel):
                continue

            bl = np.nanmean(sig[bl_mask])
            sig_bl = sig - bl  # baseline-subtracted

            mean_resp = np.nanmean(sig_bl[resp_mask])
            mean_early = np.nanmean(sig_bl[early_mask])
            mean_late = np.nanmean(sig_bl[late_mask])
            peak = np.nanmax(sig_bl[resp_mask])
            slope = mean_late - mean_early

            trials_X.append([mean_resp, mean_early, mean_late, peak, slope])
            trials_y.append(sol)

    return np.array(trials_X), np.array(trials_y)


def run_decoding(df, subject, n_folds=5, seed=42):
    """
    Run stratified k-fold decoding for one subject.
    4-way classification: water vs sucrose vs nacl vs dry.
    Also pairwise: sucrose vs water, nacl vs water.
    """
    X, y = extract_trial_features(df, subject)
    channel_id = df.loc[df["subject"]==subject, "channel_id"].iloc[0]
    pop = POP_MAP.get(channel_id, channel_id)

    results = {"subject": subject, "population": pop, "channel_id": channel_id}

    # 4-way classification
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0,
                                  solver="lbfgs", multi_class="multinomial")
        clf.fit(X_train, y[train_idx])
        y_pred = clf.predict(X_test)
        y_true_all.extend(y[test_idx])
        y_pred_all.extend(y_pred)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    results["accuracy_4way"] = balanced_accuracy_score(y_true_all, y_pred_all)
    results["confusion_4way"] = confusion_matrix(y_true_all, y_pred_all,
                                                  labels=SOL_ORDER)

    # Pairwise decodings
    for pair_name, sol_a, sol_b in [("sucrose_vs_water", "sucrose30", "water"),
                                      ("nacl_vs_water", "nacl150", "water"),
                                      ("sucrose_vs_nacl", "sucrose30", "nacl150")]:
        mask = np.isin(y, [sol_a, sol_b])
        X_pair, y_pair = X[mask], y[mask]
        if len(np.unique(y_pair)) < 2 or len(y_pair) < 10:
            results[f"accuracy_{pair_name}"] = np.nan
            continue

        skf2 = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        accs = []
        for train_idx, test_idx in skf2.split(X_pair, y_pair):
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(X_pair[train_idx])
            Xte = scaler.transform(X_pair[test_idx])
            clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0)
            clf.fit(Xtr, y_pair[train_idx])
            accs.append(balanced_accuracy_score(y_pair[test_idx], clf.predict(Xte)))
        results[f"accuracy_{pair_name}"] = np.mean(accs)

    return results


# ====================================================================
# ANALYSIS 2: EARLY VS. LATE TRIAL SPLIT
# ====================================================================

def run_split_glm(df, subject, half="first", n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
    """
    Run per-solution GLM on only the first or second half of trials within each session.
    """
    df_subj = df[df["subject"] == subject].copy()
    channel_id = df_subj["channel_id"].iloc[0]
    sessions = sorted(df_subj["blockname"].unique())

    Y_parts, trial_info_list, session_lengths = [], [], []
    time_rel, T_trial = None, None

    for sess in sessions:
        df_sess = df_subj[df_subj["blockname"] == sess].copy()
        Y_sess, ti_sess, tr, Tt, nt = get_trial_matrix(df_sess, signal_col=SIGNAL_COL)

        # Split trials
        n_half = nt // 2
        if half == "first":
            ti_keep = ti_sess[:n_half]
            Y_keep = Y_sess[:n_half * Tt]
        else:
            ti_keep = ti_sess[n_half:]
            Y_keep = Y_sess[n_half * Tt:]

        # Re-index trial_idx
        for i, t in enumerate(ti_keep):
            t["trial_idx"] = i

        Y_parts.append(Y_keep)
        trial_info_list.append(ti_keep)
        session_lengths.append(len(Y_keep))
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
    X_list, chunk_names = build_per_solution_dm(
        trial_info_list, T_trial, session_lengths, solutions=solutions)

    full_res = run_permutation_glm(
        Y, X_list, boundaries, n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=False)

    n_trials = sum(len(tl) for tl in trial_info_list)

    return {
        "subject": subject, "channel_id": channel_id,
        "half": half, "n_trials": n_trials,
        "full_glm": full_res, "chunk_names": chunk_names,
    }


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("Loading data...")
    df = load_h5_data(H5)
    subjects = sorted(df["subject"].unique())
    print(f"{len(subjects)} subjects loaded")

    # ── DECODING ──
    print("\n" + "="*70)
    print("  DECODING ANALYSIS")
    print("="*70)

    decode_results = []
    for si, subj in enumerate(subjects):
        ch = df.loc[df["subject"]==subj, "channel_id"].iloc[0]
        print(f"[{si+1}/{len(subjects)}] {subj} ({ch})...", end=" ", flush=True)
        t0 = time.time()
        res = run_decoding(df, subj)
        decode_results.append(res)
        print(f"{time.time()-t0:.1f}s  4way={res['accuracy_4way']:.3f}  "
              f"suc={res.get('accuracy_sucrose_vs_water',0):.3f}  "
              f"nacl={res.get('accuracy_nacl_vs_water',0):.3f}")

    # Summarize decoding
    dec_df = pd.DataFrame([{
        "subject": r["subject"], "population": r["population"],
        "accuracy_4way": r["accuracy_4way"],
        "accuracy_sucrose_vs_water": r.get("accuracy_sucrose_vs_water", np.nan),
        "accuracy_nacl_vs_water": r.get("accuracy_nacl_vs_water", np.nan),
        "accuracy_sucrose_vs_nacl": r.get("accuracy_sucrose_vs_nacl", np.nan),
    } for r in decode_results])

    print(f"\nDecoding summary:")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        pdf = dec_df[dec_df["population"] == pop]
        print(f"\n{pop} (n={len(pdf)}):")
        for metric in ["accuracy_4way", "accuracy_sucrose_vs_water",
                        "accuracy_nacl_vs_water", "accuracy_sucrose_vs_nacl"]:
            vals = pdf[metric].dropna()
            label = metric.replace("accuracy_", "").replace("_", " ")
            print(f"  {label:25s}  {vals.mean():.3f} +/- {vals.std():.3f}")

    # Decoding group stats
    print(f"\nGroup-level Kruskal-Wallis:")
    for metric in ["accuracy_4way", "accuracy_sucrose_vs_water",
                    "accuracy_nacl_vs_water", "accuracy_sucrose_vs_nacl"]:
        groups = [dec_df.loc[dec_df["population"]==p, metric].dropna().values
                  for p in ["Cbln4", "Crhbp", "Pnoc"]]
        if all(len(g) >= 2 for g in groups):
            H, p = kruskal(*groups)
            sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "ns"))
            label = metric.replace("accuracy_", "").replace("_", " ")
            print(f"  {label:25s}  H={H:.2f}  p={p:.4f}  {sig}")
            if p < 0.05:
                for i, j in [(0,1),(0,2),(1,2)]:
                    pn = ["Cbln4","Crhbp","Pnoc"]
                    U, pw = mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                    print(f"    {pn[i]} vs {pn[j]}: U={U:.0f}, p={pw:.4f}")

    # vs chance (0.25 for 4-way, 0.50 for pairwise)
    print(f"\nVs. chance (Wilcoxon):")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        pdf = dec_df[dec_df["population"] == pop]
        for metric, chance in [("accuracy_4way", 0.25),
                                ("accuracy_sucrose_vs_water", 0.50),
                                ("accuracy_nacl_vs_water", 0.50)]:
            vals = pdf[metric].dropna().values
            if len(vals) > 2:
                stat, pval = wilcoxon(vals - chance)
                label = metric.replace("accuracy_", "").replace("_", " ")
                sig = "*" if pval < 0.05 else "ns"
                print(f"  {pop} {label:25s}  mean={np.mean(vals):.3f}  p={pval:.4f} {sig}")

    dec_df.to_csv(SAVE_DIR / "decoding_results.csv", index=False)

    # Save confusion matrices
    conf_matrices = {}
    for r in decode_results:
        pop = r["population"]
        conf_matrices.setdefault(pop, []).append(r["confusion_4way"])

    # ── DECODING FIGURES ──
    # Figure 1: Decoding accuracy bar
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.5))
    metrics = ["accuracy_4way", "accuracy_sucrose_vs_water",
               "accuracy_nacl_vs_water", "accuracy_sucrose_vs_nacl"]
    metric_labels = ["4-way", "Sucrose vs Water", "NaCl vs Water", "Sucrose vs NaCl"]
    chances = [0.25, 0.50, 0.50, 0.50]

    for mi, (metric, label, chance) in enumerate(zip(metrics, metric_labels, chances)):
        ax = axes[mi]
        for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
            vals = dec_df.loc[dec_df["population"]==pop, metric].dropna().values
            jitter = np.random.default_rng(42+pi).normal(0, 0.08, len(vals))
            ax.scatter(np.full_like(vals, pi) + jitter, vals,
                       color=POP_COLORS[pop], s=25, alpha=0.7,
                       edgecolors="black", linewidth=0.3)
            ax.errorbar(pi, np.mean(vals), yerr=np.std(vals)/np.sqrt(len(vals)),
                        fmt="o", color="black", markersize=6, capsize=4, linewidth=1.5)
        ax.axhline(chance, color="gray", linestyle="--", linewidth=0.8, label=f"chance={chance}")
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["Cbln4", "Crhbp", "Pnoc"], fontsize=8)
        ax.set_title(label, fontsize=9, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if mi == 0:
            ax.set_ylabel("Balanced accuracy")
    fig.suptitle("Single-trial decoding accuracy", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "decoding_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: Mean confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    sol_short = ["Wat", "Suc", "NaCl", "Dry"]
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        ax = axes[pi]
        mean_cm = np.mean(conf_matrices[pop], axis=0).astype(float)
        # Normalize rows
        row_sums = mean_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = mean_cm / row_sums
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=0.7)
        ax.set_xticks(range(4))
        ax.set_xticklabels(sol_short, fontsize=8)
        ax.set_yticks(range(4))
        ax.set_yticklabels(sol_short, fontsize=8)
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if cm_norm[i,j] > 0.4 else "black")
        ax.set_title(f"{pop} (n={len(conf_matrices[pop])})", fontsize=10, fontweight="bold")
        ax.set_xlabel("Predicted")
        if pi == 0:
            ax.set_ylabel("True")
    fig.suptitle("Mean confusion matrices (row-normalized)", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "decoding_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ── EARLY VS. LATE TRIAL SPLIT ──
    print("\n\n" + "="*70)
    print("  EARLY vs. LATE TRIAL SPLIT GLM")
    print("="*70)

    split_results = {"first": [], "second": []}
    for half in ["first", "second"]:
        print(f"\n--- {half.upper()} half ---")
        for si, subj in enumerate(subjects):
            ch = df.loc[df["subject"]==subj, "channel_id"].iloc[0]
            print(f"[{si+1}/{len(subjects)}] {subj} ({ch})...", end=" ", flush=True)
            t0 = time.time()
            res = run_split_glm(df, subj, half=half, n_shifts=2000, alpha=0.01,
                                 seed=42+si)
            split_results[half].append(res)
            sig = [res["chunk_names"][k] for k in range(len(res["chunk_names"]))
                   if res["full_glm"]["significant"][k] and res["chunk_names"][k] != "session"]
            print(f"{time.time()-t0:.1f}s  R2={res['full_glm']['r2_full']:.3f}  sig: {sig if sig else 'none'}")

    # Summarize splits
    chunk_names_split = split_results["first"][0]["chunk_names"]
    task_idx_split = [i for i, n in enumerate(chunk_names_split) if n != "session"]
    sol_labels_map = {"sucrose30": "Sucrose", "nacl150": "NaCl", "dry": "Dry",
                      "lick_count": "Lick count", "trial_number": "Trial num."}

    split_df_rows = []
    for half in ["first", "second"]:
        for r in split_results[half]:
            pop = POP_MAP.get(r["channel_id"], r["channel_id"])
            for ki in task_idx_split:
                name = chunk_names_split[ki]
                split_df_rows.append({
                    "subject": r["subject"], "population": pop, "half": half,
                    "predictor": name,
                    "delta_r2": r["full_glm"]["delta_r2_obs"][ki],
                    "significant": r["full_glm"]["significant"][ki],
                    "r2_full": r["full_glm"]["r2_full"],
                })
    split_df = pd.DataFrame(split_df_rows)

    print(f"\nSplit summary:")
    for pop in ["Cbln4", "Crhbp", "Pnoc"]:
        print(f"\n{pop}:")
        for pred in ["sucrose30", "nacl150", "lick_count"]:
            for half in ["first", "second"]:
                pdf = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]==half)]
                dr2 = pdf["delta_r2"].values
                pct = 100 * pdf["significant"].mean()
                label = sol_labels_map.get(pred, pred)
                print(f"  {label:12s} {half:6s}: dR2={np.mean(dr2):.4f}+/-{np.std(dr2):.4f}  {pct:.0f}% sig")
            # Wilcoxon paired test early vs late
            early = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]=="first")].sort_values("subject")["delta_r2"].values
            late = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]=="second")].sort_values("subject")["delta_r2"].values
            if len(early) == len(late) and len(early) > 2:
                diff = late - early
                if np.any(diff != 0):
                    stat, pval = wilcoxon(diff)
                    direction = "increase" if np.median(diff) > 0 else "decrease"
                    print(f"  {'':12s} early vs late: p={pval:.4f} ({direction})")

    split_df.to_csv(SAVE_DIR / "early_late_split_results.csv", index=False)

    # ── SPLIT FIGURES ──
    # Figure 3: Early vs late dR2 comparison
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0))
    preds_to_plot = ["sucrose30", "nacl150", "lick_count"]
    pred_colors = {"sucrose30": "#2ca02c", "nacl150": "#d62728", "lick_count": "#9467bd"}

    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        ax = axes[pi]
        x = np.arange(len(preds_to_plot))
        width = 0.35
        for hi, (half, offset, alpha_val) in enumerate([("first", -width/2, 0.9), ("second", width/2, 0.5)]):
            means, sems = [], []
            for pred in preds_to_plot:
                pdf = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]==half)]
                means.append(pdf["delta_r2"].mean())
                sems.append(pdf["delta_r2"].sem())
            cols = [pred_colors[p] for p in preds_to_plot]
            bars = ax.bar(x + offset, means, width, yerr=sems,
                          color=cols, edgecolor="black", linewidth=0.5,
                          capsize=2, alpha=alpha_val,
                          label=f"{'Early' if half=='first' else 'Late'} trials")
        ax.set_xticks(x)
        ax.set_xticklabels([sol_labels_map.get(p,p) for p in preds_to_plot], rotation=45, ha="right", fontsize=8)
        n_subj = len(split_df[(split_df["population"]==pop) & (split_df["half"]=="first")]["subject"].unique())
        ax.set_title(f"{pop} (n={n_subj})", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if pi == 0:
            ax.set_ylabel("$\\Delta R^2$")
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Encoding strength: early vs. late trials", fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "early_late_split_bar.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Figure 4: Paired early-late per subject
    fig, axes = plt.subplots(3, 3, figsize=(9, 8))
    for pi, pop in enumerate(["Cbln4", "Crhbp", "Pnoc"]):
        for ki, pred in enumerate(preds_to_plot):
            ax = axes[pi, ki]
            early = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]=="first")].sort_values("subject")["delta_r2"].values
            late = split_df[(split_df["population"]==pop) & (split_df["predictor"]==pred) & (split_df["half"]=="second")].sort_values("subject")["delta_r2"].values
            for e, l in zip(early, late):
                color = "#d62728" if l < e else "#2ca02c"
                ax.plot([0, 1], [e, l], color=color, alpha=0.5, linewidth=0.8)
                ax.scatter([0, 1], [e, l], color=color, s=15, zorder=3)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Early", "Late"], fontsize=8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if ki == 0:
                ax.set_ylabel(f"{pop}\n$\\Delta R^2$", fontsize=9)
            if pi == 0:
                ax.set_title(sol_labels_map.get(pred, pred), fontsize=10, fontweight="bold")
            # Add p-value
            if len(early) == len(late) and len(early) > 2:
                diff = late - early
                if np.any(diff != 0):
                    _, pval = wilcoxon(diff)
                    ax.text(0.5, 0.95, f"p={pval:.3f}", transform=ax.transAxes,
                            ha="center", va="top", fontsize=7,
                            fontweight="bold" if pval < 0.05 else "normal")
    fig.suptitle("Early vs. late trials: paired subject comparisons",
                 fontsize=11, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "early_late_split_paired.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nAll figures and CSVs saved. Done.")
