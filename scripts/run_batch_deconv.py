"""Run group analysis on deconvolved dF/F signal, in two batches."""
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
SIGNAL_COL = "delta_signal_poly_dff_deconv"

def run_subject_glm_deconv(df, subject, n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
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
    X_list, chunk_names = build_design_matrix(
        trial_info_list, T_trial, session_lengths, solutions=solutions
    )

    full_res = run_permutation_glm(
        Y, X_list, boundaries, n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=False
    )
    tr_res = run_timeresolved(
        Y, trial_info_list, time_rel, T_trial, session_trial_counts,
        win_size=1.0, win_step=0.25, n_shifts=n_shifts, alpha=alpha,
        seed=seed, verbose=False
    )
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

print(f"Batch {batch}: {len(batch_subjects)} subjects, signal={SIGNAL_COL}")
results = []
t0 = time.time()
for si, subj in enumerate(batch_subjects):
    ch = df[df["subject"] == subj]["channel_id"].iloc[0]
    t_subj = time.time()
    print(f"[{si+1}/{len(batch_subjects)}] {subj} ({ch})...", end=" ", flush=True)
    res = run_subject_glm_deconv(df, subj, n_shifts=5000, alpha=0.01,
                                  seed=42+si+(batch-1)*16)
    results.append(res)
    elapsed = time.time() - t_subj
    sig = [res["chunk_names"][k] for k in range(len(res["chunk_names"]))
           if res["full_glm"]["significant"][k] and res["chunk_names"][k] != "session"]
    print(f"{elapsed:.1f}s  R2={res['full_glm']['r2_full']:.3f}  sig: {sig if sig else 'none'}")

print(f"\nBatch {batch} done in {time.time()-t0:.0f}s")
with open(SAVE_DIR / f"_deconv_batch{batch}_results.pkl", "wb") as f:
    pickle.dump(results, f)
print(f"Saved to _deconv_batch{batch}_results.pkl")
