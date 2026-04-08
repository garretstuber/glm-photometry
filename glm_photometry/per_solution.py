"""
Run group GLM with INDIVIDUAL SOLUTION CONTRASTS.

Instead of one omnibus 'solution' predictor (K-1 dummy columns tested jointly),
each non-reference solution gets its own single-column predictor chunk. This
yields a separate delta-R2 and p-value for sucrose vs water, NaCl vs water,
and dry vs water.

Design matrix chunks:
  0: sucrose   (1 column, dummy: sucrose30 vs water)
  1: nacl      (1 column, dummy: nacl150 vs water)
  2: dry       (1 column, dummy: dry vs water)
  3: lick_count (1 column, z-scored)
  4: trial_number (1 column, linear drift)
  5: session   (n_sessions - 1 columns)

Two batches to stay within timeout.
"""
import sys, time, pickle
from pathlib import Path
import numpy as np

from .group_analysis import load_h5_data
from .core import (
    get_trial_matrix, run_permutation_glm,
    run_timeresolved, N_SHIFTS, ALPHA,
)
SIGNAL_COL = "delta_signal_poly_zscore_blsub"
REFERENCE = "water"


def build_per_solution_dm(trial_info_list, T_trial, session_lengths, solutions=None):
    """
    Build design matrix with each non-reference solution as a separate chunk.

    Returns
    -------
    X_list : list of arrays, one per chunk
    chunk_names : list of str
    """
    if solutions is None:
        all_sols = set()
        for tl in trial_info_list:
            for t in tl:
                all_sols.add(t["solution"])
        solutions = sorted(all_sols)

    non_ref = [s for s in solutions if s != REFERENCE]
    n_sessions = len(trial_info_list)
    T_total = sum(session_lengths)

    # Per-solution dummy columns (each is a separate chunk)
    sol_blocks = {s: np.zeros((T_total, 1), dtype=np.float64) for s in non_ref}

    # Lick count
    lick_block = np.zeros((T_total, 1), dtype=np.float64)

    # Trial number
    trial_block = np.zeros((T_total, 1), dtype=np.float64)

    # Session dummies
    sess_block = np.zeros((T_total, max(n_sessions - 1, 1)), dtype=np.float64)

    # Collect all lick counts for z-scoring
    all_licks = []
    for tl in trial_info_list:
        for t in tl:
            all_licks.append(t["lick_count"])
    lick_mean = np.mean(all_licks)
    lick_std = np.std(all_licks)
    if lick_std == 0:
        lick_std = 1.0

    offset = 0
    for si, trial_info in enumerate(trial_info_list):
        n_trials = len(trial_info)
        for ti, tdict in enumerate(trial_info):
            sl = slice(offset + ti * T_trial, offset + (ti + 1) * T_trial)
            sol = tdict["solution"]

            # Solution dummies
            if sol in sol_blocks:
                sol_blocks[sol][sl, 0] = 1.0

            # Lick count (z-scored)
            lick_z = (tdict["lick_count"] - lick_mean) / lick_std
            lick_block[sl, 0] = lick_z

            # Trial number (normalized 0-1)
            trial_frac = ti / max(n_trials - 1, 1)
            trial_block[sl, 0] = trial_frac

        # Session dummies
        if si > 0 and n_sessions > 1:
            sess_sl = slice(offset, offset + session_lengths[si])
            sess_block[sess_sl, si - 1] = 1.0

        offset += session_lengths[si]

    # Assemble: each solution as its own chunk
    X_list = [sol_blocks[s] for s in non_ref] + [lick_block, trial_block, sess_block]
    chunk_names = non_ref + ["lick_count", "trial_number", "session"]

    return X_list, chunk_names


def build_per_solution_windowed_dm(trial_info_list, T_trial, time_rel,
                                    win_start, win_end, session_lengths_full,
                                    solutions=None):
    """Windowed version of per-solution DM for time-resolved analysis."""
    if solutions is None:
        all_sols = set()
        for tl in trial_info_list:
            for t in tl:
                all_sols.add(t["solution"])
        solutions = sorted(all_sols)

    non_ref = [s for s in solutions if s != REFERENCE]
    win_mask = (time_rel >= win_start) & (time_rel < win_end)
    T_win = int(win_mask.sum())
    if T_win == 0:
        return None, None, None, None, None

    n_sessions = len(trial_info_list)
    session_lengths_win = []
    for si, tl in enumerate(trial_info_list):
        session_lengths_win.append(len(tl) * T_win)
    T_total_win = sum(session_lengths_win)

    sol_blocks = {s: np.zeros((T_total_win, 1), dtype=np.float64) for s in non_ref}
    lick_block = np.zeros((T_total_win, 1), dtype=np.float64)
    trial_block = np.zeros((T_total_win, 1), dtype=np.float64)
    sess_block = np.zeros((T_total_win, max(n_sessions - 1, 1)), dtype=np.float64)

    all_licks = [t["lick_count"] for tl in trial_info_list for t in tl]
    lick_mean, lick_std = np.mean(all_licks), np.std(all_licks)
    if lick_std == 0:
        lick_std = 1.0

    offset = 0
    for si, trial_info in enumerate(trial_info_list):
        n_trials = len(trial_info)
        for ti, tdict in enumerate(trial_info):
            sl = slice(offset + ti * T_win, offset + (ti + 1) * T_win)
            sol = tdict["solution"]
            if sol in sol_blocks:
                sol_blocks[sol][sl, 0] = 1.0
            lick_block[sl, 0] = (tdict["lick_count"] - lick_mean) / lick_std
            trial_block[sl, 0] = ti / max(n_trials - 1, 1)
        if si > 0 and n_sessions > 1:
            sess_sl = slice(offset, offset + session_lengths_win[si])
            sess_block[sess_sl, si - 1] = 1.0
        offset += session_lengths_win[si]

    X_list = [sol_blocks[s] for s in non_ref] + [lick_block, trial_block, sess_block]
    chunk_names = non_ref + ["lick_count", "trial_number", "session"]

    boundaries_win = []
    off = 0
    for sl in session_lengths_win:
        boundaries_win.append((off, off + sl))
        off += sl

    return X_list, chunk_names, boundaries_win, session_lengths_win, T_win


def run_timeresolved_per_sol(Y, trial_info_list, time_rel, T_trial,
                              session_trial_counts, solutions,
                              win_size=1.0, win_step=0.25,
                              n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
    """Time-resolved GLM with per-solution predictors."""
    t_min, t_max = time_rel.min(), time_rel.max()
    win_starts = np.arange(t_min, t_max - win_size + win_step, win_step)
    win_centers = win_starts + win_size / 2
    session_lengths_full = [c * T_trial for c in session_trial_counts]

    # Get K from test
    test = build_per_solution_windowed_dm(
        trial_info_list, T_trial, time_rel,
        win_starts[0], win_starts[0] + win_size,
        session_lengths_full, solutions=solutions)
    K = len(test[0])
    chunk_names = test[1]
    n_windows = len(win_starts)

    all_pvals = np.ones((n_windows, K))
    all_dr2 = np.zeros((n_windows, K))
    all_sig = np.zeros((n_windows, K), dtype=bool)

    for wi, ws in enumerate(win_starts):
        we = ws + win_size
        win_mask = (time_rel >= ws) & (time_rel < we)
        win_idx = np.where(win_mask)[0]
        T_win = len(win_idx)
        if T_win == 0:
            continue

        # Build windowed Y
        Y_win_parts = []
        offset = 0
        for si, trial_info in enumerate(trial_info_list):
            n_trials = len(trial_info)
            for ti in range(n_trials):
                trial_start = offset + ti * T_trial
                Y_win_parts.append(Y[trial_start + win_idx[0]:trial_start + win_idx[0] + T_win])
            offset += n_trials * T_trial
        Y_win = np.concatenate(Y_win_parts)

        X_win, _, bounds_win, _, _ = build_per_solution_windowed_dm(
            trial_info_list, T_trial, time_rel, ws, we,
            session_lengths_full, solutions=solutions)
        if X_win is None:
            continue

        res = run_permutation_glm(
            Y_win, X_win, bounds_win,
            n_shifts=n_shifts, alpha=alpha, seed=seed+wi, verbose=False)

        all_pvals[wi, :] = res["pvals"]
        all_dr2[wi, :] = res["delta_r2_obs"]
        all_sig[wi, :] = res["significant"]

    return {
        "win_centers": win_centers,
        "pvals": all_pvals, "delta_r2": all_dr2, "significant": all_sig,
        "chunk_names": chunk_names,
    }


def run_subject_glm_per_sol(df, subject, n_shifts=N_SHIFTS, alpha=ALPHA, seed=42):
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
    X_list, chunk_names = build_per_solution_dm(
        trial_info_list, T_trial, session_lengths, solutions=solutions)

    full_res = run_permutation_glm(
        Y, X_list, boundaries, n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=False)

    tr_res = run_timeresolved_per_sol(
        Y, trial_info_list, time_rel, T_trial, session_trial_counts,
        solutions=solutions, win_size=1.0, win_step=0.25,
        n_shifts=n_shifts, alpha=alpha, seed=seed)

    return {
        "subject": subject, "channel_id": channel_id,
        "n_sessions": len(sessions),
        "n_trials": sum(session_trial_counts),
        "full_glm": full_res, "timeresolved": tr_res,
        "chunk_names": chunk_names, "time_rel": time_rel,
    }


