"""
Run group GLM analysis WITHOUT the lick x solution interaction term.
Design matrix: solution + lick_count + trial_number + session (4 chunks).
Two batches to stay within timeout.
"""
import sys, time, pickle
from pathlib import Path

sys.path.insert(0, str(Path("/sessions/dazzling-vigilant-fermat/mnt/20260331 - Elerding et al., VTA GABA neurons")))
from fp_glm_group_analysis import load_h5_data
from fp_glm_circular_shift import (
    get_trial_matrix, build_design_matrix, run_permutation_glm,
    run_timeresolved, N_SHIFTS, ALPHA,
)
import numpy as np

SAVE_DIR = Path("/sessions/dazzling-vigilant-fermat/mnt/20260331 - Elerding et al., VTA GABA neurons")
H5 = SAVE_DIR / "streams_peth_multi_mix_all.h5"
SIGNAL_COL = "delta_signal_poly_zscore_blsub"


def run_subject_glm_no_lxs(df, subject, n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
    df_subj = df[df["subject"] == subject].copy()
    channel_id = df_subj["channel_id"].iloc[0]
    sessions = sorted(df_subj["blockname"].unique())

    Y_parts, trial_info_list, session_lengths, session_trial_counts = [], [], [], []
    time_rel, T_trial = None, None

    for sess in sessions:
        df_sess = df_subj[df_subj["blockname"] == sess].copy()
        Y_sess, ti_sess, tr, Tt, nt = get_trial_matrix(df_sess, signal_col=SIGNAL_COL)
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
    X_list_full, chunk_names_full = build_design_matrix(
        trial_info_list, T_trial, session_lengths, solutions=solutions
    )

    # Remove lick_x_solution (index 2) from X_list and chunk_names
    drop_idx = chunk_names_full.index("lick_x_solution")
    X_list = [X for i, X in enumerate(X_list_full) if i != drop_idx]
    chunk_names = [n for i, n in enumerate(chunk_names_full) if i != drop_idx]

    full_res = run_permutation_glm(
        Y, X_list, boundaries, n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=False
    )

    # Time-resolved GLM needs a custom wrapper since run_timeresolved
    # calls build_windowed_dm internally. We'll run it with the full model
    # and then extract only the non-interaction chunks from the results.
    # Actually, cleaner to just run the permutation on each window manually.
    # But that's a big rewrite. Instead, let's use a monkey-patch approach:
    # run the full time-resolved, then drop the lick_x_solution column from results.
    from fp_glm_circular_shift import run_timeresolved as _run_tr
    tr_res_full = _run_tr(
        Y, trial_info_list, time_rel, T_trial, session_trial_counts,
        win_size=1.0, win_step=0.25, n_shifts=n_shifts, alpha=alpha,
        seed=seed, verbose=False
    )

    # Drop the lick_x_solution index from time-resolved results
    full_chunk_names_tr = tr_res_full["chunk_names"]
    drop_idx_tr = full_chunk_names_tr.index("lick_x_solution")
    keep_idx = [i for i in range(len(full_chunk_names_tr)) if i != drop_idx_tr]

    tr_res = {
        "win_centers": tr_res_full["win_centers"],
        "pvals": tr_res_full["pvals"][:, keep_idx],
        "delta_r2": tr_res_full["delta_r2"][:, keep_idx],
        "significant": tr_res_full["significant"][:, keep_idx],
        "chunk_names": [full_chunk_names_tr[i] for i in keep_idx],
    }

    return {
        "subject": subject, "channel_id": channel_id,
        "n_sessions": len(sessions),
        "n_trials": sum(session_trial_counts),
        "full_glm": full_res, "timeresolved": tr_res,
        "chunk_names": chunk_names, "time_rel": time_rel,
    }


batch = int(sys.argv[1])
print(f"Loading data...")
df = load_h5_data(H5)
subjects = sorted(df["subject"].unique())
batch_subjects = subjects[:16] if batch == 1 else subjects[16:]

print(f"Batch {batch}: {len(batch_subjects)} subjects, NO lick_x_solution interaction")
results = []
t0 = time.time()
for si, subj in enumerate(batch_subjects):
    ch = df[df["subject"] == subj]["channel_id"].iloc[0]
    t_subj = time.time()
    print(f"[{si+1}/{len(batch_subjects)}] {subj} ({ch})...", end=" ", flush=True)
    res = run_subject_glm_no_lxs(df, subj, n_shifts=5000, alpha=0.01,
                                  seed=42+si+(batch-1)*16)
    results.append(res)
    elapsed = time.time() - t_subj
    sig = [res["chunk_names"][k] for k in range(len(res["chunk_names"]))
           if res["full_glm"]["significant"][k] and res["chunk_names"][k] != "session"]
    print(f"{elapsed:.1f}s  R2={res['full_glm']['r2_full']:.3f}  sig: {sig if sig else 'none'}")

print(f"\nBatch {batch} done in {time.time()-t0:.0f}s")
with open(SAVE_DIR / f"_no_lxs_batch{batch}_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"Saved to _no_lxs_batch{batch}_results.pkl")
