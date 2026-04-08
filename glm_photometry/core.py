"""
Circular-shift permutation GLM for fiber photometry peri-event data.

Adapted from the calcium imaging GLM (Hjort et al. / Stuber lab) for
single-channel fiber photometry recordings. Key differences from the
imaging version:
  - Single channel per recording (not hundreds of neurons)
  - Predictors: solution identity, lick count, lick x solution, trial number
  - 5,000 circular-shift permutations (richer null for single-channel data)
  - Data comes from a tidy DataFrame rather than HDF5

Garret Stuber lab, UW -- Elerding et al., VTA GABA neurons
"""

import numpy as np
import pandas as pd
from scipy.linalg import svd
from pathlib import Path


# ====================================================================
# CONSTANTS
# ====================================================================

ALPHA = 0.01
N_SHIFTS = 5000
REFERENCE_SOLUTION = "water"  # dummy-coding reference


# ====================================================================
# DATA LOADER
# ====================================================================

def load_fp_data(xlsx_path):
    """
    Load fiber photometry peri-event DataFrame from Excel.

    Returns
    -------
    df : pd.DataFrame with columns:
        blockname, subject, channel_id, procedure, time, time_rel,
        event_id_char, event_number_solution, solution, lick_count,
        delta_signal_poly_zscore_blsub
    """
    df = pd.read_excel(xlsx_path)
    # Sort for consistent ordering
    df = df.sort_values(
        ["channel_id", "subject", "blockname", "solution",
         "event_number_solution", "time_rel"]
    ).reset_index(drop=True)
    return df


def get_trial_matrix(df_session, signal_col="delta_signal_poly_zscore_blsub"):
    """
    Reshape a single session/channel subset into a trial matrix.

    Parameters
    ----------
    df_session : DataFrame
        Subset for one blockname + channel_id combination.
    signal_col : str
        Column name for the photometry signal to use as the response variable.

    Returns
    -------
    Y : (n_trials * T_trial,) array -- concatenated signal
    trial_info : list of dicts with keys:
        solution, lick_count, event_number_solution, trial_idx
    time_rel : (T_trial,) array -- time axis relative to event onset
    T_trial : int -- timepoints per trial
    n_trials : int
    """
    # Get the time axis from the first trial
    first_sol = df_session["solution"].iloc[0]
    first_evt = df_session["event_number_solution"].iloc[0]
    mask = (
        (df_session["solution"] == first_sol)
        & (df_session["event_number_solution"] == first_evt)
    )
    time_rel = df_session.loc[mask, "time_rel"].values
    T_trial = len(time_rel)

    # Enumerate all trials in presentation order (by absolute time)
    trial_keys = (
        df_session.groupby(["solution", "event_number_solution"])["time"]
        .min()
        .reset_index()
        .sort_values("time")
    )

    n_trials = len(trial_keys)
    Y = np.zeros(n_trials * T_trial, dtype=np.float64)
    trial_info = []

    for ti, (_, row) in enumerate(trial_keys.iterrows()):
        sol = row["solution"]
        evt = row["event_number_solution"]
        mask = (
            (df_session["solution"] == sol)
            & (df_session["event_number_solution"] == evt)
        )
        sig = df_session.loc[mask, signal_col].values
        lick = df_session.loc[mask, "lick_count"].values[0]

        # Handle length mismatches (should not happen, but be safe)
        n = min(len(sig), T_trial)
        Y[ti * T_trial: ti * T_trial + n] = sig[:n]

        trial_info.append({
            "solution": sol,
            "lick_count": lick,
            "event_number_solution": evt,
            "trial_idx": ti,
        })

    return Y, trial_info, time_rel, T_trial, n_trials


# ====================================================================
# DESIGN MATRIX CONSTRUCTION
# ====================================================================

def build_design_matrix(trial_info_list, T_trial, session_lengths,
                        solutions=None):
    """
    Build the full concatenated design matrix for one channel across sessions.

    Predictor chunks (0-indexed):
      0: solution  -- dummy-coded (K-1 columns, water = reference)
      1: lick_count -- z-scored continuous (1 column)
      2: lick_x_solution -- lick_count * each solution dummy (K-1 columns)
      3: trial_number -- linear drift within session (1 column)
      4: session -- session indicator dummies (n_sessions - 1 columns)

    Parameters
    ----------
    trial_info_list : list of list-of-dicts
        One inner list per session, each element is a trial dict.
    T_trial : int
        Timepoints per trial.
    session_lengths : list of int
        Number of total timepoints per session (n_trials * T_trial).
    solutions : list of str or None
        Ordered solution names. Inferred from data if None.

    Returns
    -------
    X_list : list of (T_total, n_cols_k) arrays, one per predictor chunk
    chunk_names : list of str
    """
    if solutions is None:
        all_sols = set()
        for tl in trial_info_list:
            for t in tl:
                all_sols.add(t["solution"])
        solutions = sorted(all_sols)

    # Non-reference solutions for dummy coding
    non_ref = [s for s in solutions if s != REFERENCE_SOLUTION]
    n_dummies = len(non_ref)

    # Collect all lick counts for z-scoring
    all_licks = []
    for tl in trial_info_list:
        for t in tl:
            all_licks.append(t["lick_count"])
    lick_mean = np.mean(all_licks)
    lick_std = np.std(all_licks)
    if lick_std == 0:
        lick_std = 1.0

    n_sessions = len(trial_info_list)
    T_total = sum(session_lengths)

    # Initialize chunks
    sol_block = np.zeros((T_total, n_dummies), dtype=np.float64)
    lick_block = np.zeros((T_total, 1), dtype=np.float64)
    lxs_block = np.zeros((T_total, n_dummies), dtype=np.float64)
    trial_block = np.zeros((T_total, 1), dtype=np.float64)
    sess_block = np.zeros((T_total, max(1, n_sessions - 1)), dtype=np.float64)

    offset = 0
    for si, trial_info in enumerate(trial_info_list):
        n_trials = len(trial_info)
        for ti, t in enumerate(trial_info):
            sl = slice(offset + ti * T_trial, offset + (ti + 1) * T_trial)

            # Solution dummies
            for di, sol_name in enumerate(non_ref):
                if t["solution"] == sol_name:
                    sol_block[sl, di] = 1.0

            # Lick count (z-scored)
            lick_z = (t["lick_count"] - lick_mean) / lick_std
            lick_block[sl, 0] = lick_z

            # Lick x solution interaction
            for di, sol_name in enumerate(non_ref):
                if t["solution"] == sol_name:
                    lxs_block[sl, di] = lick_z

            # Trial number (linear drift within session)
            trial_frac = ti / max(n_trials - 1, 1)
            trial_block[sl, 0] = trial_frac

        # Session indicator
        if si > 0 and n_sessions > 1:
            sess_sl = slice(offset, offset + session_lengths[si])
            sess_block[sess_sl, si - 1] = 1.0

        offset += session_lengths[si]

    X_list = [sol_block, lick_block, lxs_block, trial_block, sess_block]
    chunk_names = ["solution", "lick_count", "lick_x_solution",
                   "trial_number", "session"]
    return X_list, chunk_names


def build_windowed_dm(trial_info_list, T_trial, time_rel,
                      win_start, win_end, session_lengths_full,
                      solutions=None):
    """
    Build design matrix restricted to [win_start, win_end] of each trial.

    Same predictor chunks as build_design_matrix but only uses timepoints
    within the specified window relative to event onset.

    Returns
    -------
    X_list, chunk_names, Y_win, boundaries_win, T_win
    """
    win_mask = (time_rel >= win_start) & (time_rel < win_end)
    win_idx = np.where(win_mask)[0]
    T_win = len(win_idx)
    if T_win == 0:
        return None, None, None, None, 0

    if solutions is None:
        all_sols = set()
        for tl in trial_info_list:
            for t in tl:
                all_sols.add(t["solution"])
        solutions = sorted(all_sols)

    non_ref = [s for s in solutions if s != REFERENCE_SOLUTION]
    n_dummies = len(non_ref)

    all_licks = []
    for tl in trial_info_list:
        for t in tl:
            all_licks.append(t["lick_count"])
    lick_mean = np.mean(all_licks)
    lick_std = np.std(all_licks) if np.std(all_licks) > 0 else 1.0

    n_sessions = len(trial_info_list)
    session_lengths_win = [len(tl) * T_win for tl in trial_info_list]
    T_total_win = sum(session_lengths_win)

    sol_block = np.zeros((T_total_win, n_dummies), dtype=np.float64)
    lick_block = np.zeros((T_total_win, 1), dtype=np.float64)
    lxs_block = np.zeros((T_total_win, n_dummies), dtype=np.float64)
    trial_block = np.zeros((T_total_win, 1), dtype=np.float64)
    sess_block = np.zeros((T_total_win, max(1, n_sessions - 1)),
                          dtype=np.float64)

    offset = 0
    for si, trial_info in enumerate(trial_info_list):
        n_trials = len(trial_info)
        for ti, t in enumerate(trial_info):
            sl = slice(offset + ti * T_win, offset + (ti + 1) * T_win)
            for di, sol_name in enumerate(non_ref):
                if t["solution"] == sol_name:
                    sol_block[sl, di] = 1.0
            lick_z = (t["lick_count"] - lick_mean) / lick_std
            lick_block[sl, 0] = lick_z
            for di, sol_name in enumerate(non_ref):
                if t["solution"] == sol_name:
                    lxs_block[sl, di] = lick_z
            trial_frac = ti / max(n_trials - 1, 1)
            trial_block[sl, 0] = trial_frac
        if si > 0 and n_sessions > 1:
            sess_sl = slice(offset, offset + session_lengths_win[si])
            sess_block[sess_sl, si - 1] = 1.0
        offset += session_lengths_win[si]

    X_list = [sol_block, lick_block, lxs_block, trial_block, sess_block]
    chunk_names = ["solution", "lick_count", "lick_x_solution",
                   "trial_number", "session"]
    boundaries_win = []
    off = 0
    for sl in session_lengths_win:
        boundaries_win.append((off, off + sl))
        off += sl

    return X_list, chunk_names, boundaries_win, session_lengths_win, T_win


# ====================================================================
# CIRCULAR-SHIFT PERMUTATION ENGINE
# ====================================================================

def circular_shift_1d(y, boundaries, rng):
    """
    Circularly shift a 1D signal within session boundaries.

    Parameters
    ----------
    y : (T,) array
    boundaries : list of (start, end) tuples
    rng : numpy random Generator

    Returns
    -------
    y_null : (T,) array
    """
    y_null = np.empty_like(y)
    for start, end in boundaries:
        T_block = end - start
        q = rng.integers(1, T_block)
        block = y[start:end]
        y_null[start:end] = np.concatenate([block[q:], block[:q]])
    return y_null


def svd_setup(X_list, T):
    """
    Pre-compute SVD components for full and reduced models.

    Returns
    -------
    U_full : left singular vectors of full model
    U_reds : list of left singular vectors for each leave-one-out model
    m_ks : list of column counts per chunk
    M : total columns in full model (including intercept)
    """
    X_full = np.ones((T, 1), dtype=np.float64)
    for Xc in X_list:
        X_full = np.hstack((X_full, Xc))
    M = X_full.shape[1]

    # Remove near-zero variance columns (rank deficiency)
    col_var = np.var(X_full, axis=0)
    keep = col_var > 1e-10
    keep[0] = True  # always keep intercept
    X_full_clean = X_full[:, keep]
    M = X_full_clean.shape[1]
    U_full = svd(X_full_clean, full_matrices=False)[0]

    K = len(X_list)
    U_reds = []
    m_ks = []
    for k in range(K):
        m_k = X_list[k].shape[1]
        m_ks.append(m_k)
        X_red = np.ones((T, 1), dtype=np.float64)
        for kt in range(K):
            if kt != k:
                X_red = np.hstack((X_red, X_list[kt]))
        var_red = np.var(X_red, axis=0)
        keep_red = var_red > 1e-10
        keep_red[0] = True
        X_red_clean = X_red[:, keep_red]
        U_reds.append(svd(X_red_clean, full_matrices=False)[0])

    return U_full, U_reds, m_ks, M


def compute_fstats_1d(y, U_full, U_reds, m_ks, M, T):
    """
    Compute leave-one-out F-statistics for a single signal.

    Returns
    -------
    f_vals : (K,) array of F-statistics
    delta_r2 : (K,) array of unique variance explained
    """
    K = len(U_reds)
    y = y.astype(np.float64)
    y_sq = np.sum(y ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)

    rss_full = y_sq - np.sum((U_full.T @ y) ** 2)
    rss_full = max(rss_full, 1e-30)  # prevent division by zero
    r2_full = 1.0 - rss_full / ss_tot if ss_tot > 0 else 0.0

    f_vals = np.zeros(K)
    delta_r2 = np.zeros(K)
    for k in range(K):
        rss_red = y_sq - np.sum((U_reds[k].T @ y) ** 2)
        denom = rss_full / (T - M)
        if denom > 0 and m_ks[k] > 0:
            f_vals[k] = ((rss_red - rss_full) / m_ks[k]) / denom
        if ss_tot > 0:
            r2_red = 1.0 - rss_red / ss_tot
            delta_r2[k] = r2_full - r2_red

    return f_vals, delta_r2


def run_permutation_glm(Y, X_list, boundaries, n_shifts=N_SHIFTS,
                        alpha=ALPHA, seed=42, verbose=True):
    """
    Full circular-shift permutation GLM for a single FP channel.

    Parameters
    ----------
    Y : (T,) array -- concatenated photometry signal
    X_list : list of (T, n_cols_k) arrays -- predictor chunks
    boundaries : list of (start, end) tuples -- session boundaries
    n_shifts : int -- number of permutations
    alpha : float -- significance threshold
    seed : int
    verbose : bool

    Returns
    -------
    results : dict with keys:
        f_obs : (K,) observed F-statistics
        delta_r2_obs : (K,) observed delta-R2
        f_null : (n_shifts, K) null F-statistics
        pvals : (K,) p-values
        significant : (K,) bool array
        r2_full : float -- full model R2
    """
    T = len(Y)
    K = len(X_list)
    rng = np.random.default_rng(seed=seed)

    # SVD setup
    U_full, U_reds, m_ks, M = svd_setup(X_list, T)

    # Observed statistics
    f_obs, delta_r2_obs = compute_fstats_1d(Y, U_full, U_reds, m_ks, M, T)

    # Full model R2
    y64 = Y.astype(np.float64)
    ss_tot = np.sum((y64 - np.mean(y64)) ** 2)
    rss_full = np.sum(y64 ** 2) - np.sum((U_full.T @ y64) ** 2)
    r2_full = 1.0 - rss_full / ss_tot if ss_tot > 0 else 0.0

    # Null distribution
    f_null = np.zeros((n_shifts, K), dtype=np.float64)
    for i in range(n_shifts):
        if verbose and (i + 1) % 500 == 0:
            print(f"  shift {i+1}/{n_shifts}")
        Y_shifted = circular_shift_1d(Y, boundaries, rng)
        f_null[i, :], _ = compute_fstats_1d(
            Y_shifted, U_full, U_reds, m_ks, M, T
        )

    # P-values (conservative +1 correction)
    pvals = np.zeros(K)
    for k in range(K):
        pvals[k] = (np.sum(f_null[:, k] >= f_obs[k]) + 1) / (n_shifts + 1)

    significant = pvals < alpha

    return {
        "f_obs": f_obs,
        "delta_r2_obs": delta_r2_obs,
        "f_null": f_null,
        "pvals": pvals,
        "significant": significant,
        "r2_full": r2_full,
    }


# ====================================================================
# TIME-RESOLVED WINDOWED GLM
# ====================================================================

def run_timeresolved(Y_full_trials, trial_info_list, time_rel, T_trial,
                     session_trial_counts, win_size=1.0, win_step=0.25,
                     n_shifts=N_SHIFTS, alpha=ALPHA, seed=42,
                     verbose=True):
    """
    Slide a window across the peri-event period and run the GLM at each step.

    Parameters
    ----------
    Y_full_trials : (T_total,) array -- full concatenated signal
    trial_info_list : list of list-of-dicts
    time_rel : (T_trial,) array
    T_trial : int
    session_trial_counts : list of int (trials per session)
    win_size, win_step : float (seconds)
    n_shifts : int
    alpha : float

    Returns
    -------
    results : dict with keys:
        win_centers : array of window centers
        pvals : (n_windows, K) array
        delta_r2 : (n_windows, K) array
        significant : (n_windows, K) bool
        chunk_names : list of str
    """
    t_min = time_rel.min()
    t_max = time_rel.max()
    win_starts = np.arange(t_min, t_max - win_size + win_step, win_step)
    win_centers = win_starts + win_size / 2

    solutions = sorted(set(
        t["solution"] for tl in trial_info_list for t in tl
    ))

    n_windows = len(win_starts)
    # Determine K from a test build
    test_out = build_windowed_dm(
        trial_info_list, T_trial, time_rel,
        win_starts[0], win_starts[0] + win_size,
        [c * T_trial for c in session_trial_counts],
        solutions=solutions,
    )
    K = len(test_out[0])
    chunk_names = test_out[1]

    all_pvals = np.ones((n_windows, K))
    all_dr2 = np.zeros((n_windows, K))
    all_sig = np.zeros((n_windows, K), dtype=bool)

    for wi, ws in enumerate(win_starts):
        we = ws + win_size
        if verbose:
            print(f"Window {wi+1}/{n_windows}: [{ws:.2f}, {we:.2f})")

        # Extract windowed signal
        win_mask = (time_rel >= ws) & (time_rel < we)
        win_idx = np.where(win_mask)[0]
        T_win = len(win_idx)
        if T_win == 0:
            continue

        # Build windowed Y
        n_sessions = len(trial_info_list)
        Y_win_parts = []
        offset = 0
        for si, trial_info in enumerate(trial_info_list):
            n_trials = len(trial_info)
            for ti in range(n_trials):
                trial_start = offset + ti * T_trial
                Y_win_parts.append(
                    Y_full_trials[trial_start + win_idx[0]:
                                  trial_start + win_idx[0] + T_win]
                )
            offset += n_trials * T_trial
        Y_win = np.concatenate(Y_win_parts)

        # Build windowed design matrix
        session_lengths_full = [c * T_trial for c in session_trial_counts]
        X_win, _, bounds_win, _, _ = build_windowed_dm(
            trial_info_list, T_trial, time_rel, ws, we,
            session_lengths_full, solutions=solutions,
        )
        if X_win is None:
            continue

        # Run permutation GLM
        res = run_permutation_glm(
            Y_win, X_win, bounds_win,
            n_shifts=n_shifts, alpha=alpha,
            seed=seed + wi, verbose=False,
        )

        all_pvals[wi, :] = res["pvals"]
        all_dr2[wi, :] = res["delta_r2_obs"]
        all_sig[wi, :] = res["significant"]

    return {
        "win_centers": win_centers,
        "pvals": all_pvals,
        "delta_r2": all_dr2,
        "significant": all_sig,
        "chunk_names": chunk_names,
    }


# ====================================================================
# MAIN ANALYSIS RUNNER
# ====================================================================

def prepare_channel_data(df, channel_id, signal_col="delta_signal_poly_zscore_blsub"):
    """
    Prepare concatenated signal and trial info for one channel_id.

    Groups by blockname (session), concatenates trials in presentation
    order within each session.

    Returns
    -------
    Y : (T_total,) array
    trial_info_list : list of list-of-dicts (one per session)
    time_rel : (T_trial,) array
    T_trial : int
    session_lengths : list of int (total timepoints per session)
    session_trial_counts : list of int (trials per session)
    boundaries : list of (start, end) tuples
    sessions : list of str (blockname values)
    """
    df_ch = df[df["channel_id"] == channel_id].copy()
    sessions = sorted(df_ch["blockname"].unique())

    Y_parts = []
    trial_info_list = []
    session_lengths = []
    session_trial_counts = []
    time_rel = None
    T_trial = None

    for sess in sessions:
        df_sess = df_ch[df_ch["blockname"] == sess].copy()
        Y_sess, ti_sess, tr, Tt, nt = get_trial_matrix(df_sess, signal_col=signal_col)
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

    return (Y, trial_info_list, time_rel, T_trial,
            session_lengths, session_trial_counts, boundaries, sessions)


def run_full_analysis(df, channel_id, n_shifts=N_SHIFTS, alpha=ALPHA,
                      seed=42, verbose=True):
    """
    Run the complete GLM analysis for one channel_id.

    Returns
    -------
    results : dict with keys:
        channel_id : str
        full_glm : dict from run_permutation_glm
        timeresolved : dict from run_timeresolved
        chunk_names : list of str
        meta : dict with session info
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Channel: {channel_id}")
        print(f"{'='*60}")

    (Y, trial_info_list, time_rel, T_trial,
     session_lengths, session_trial_counts,
     boundaries, sessions) = prepare_channel_data(df, channel_id)

    if verbose:
        n_total_trials = sum(session_trial_counts)
        print(f"Sessions: {sessions}")
        print(f"Total trials: {n_total_trials}")
        print(f"Timepoints per trial: {T_trial}")
        print(f"Total timepoints: {len(Y)}")
        print(f"\nRunning full-trial GLM ({n_shifts} permutations)...")

    # Build design matrix
    solutions = sorted(df["solution"].unique())
    X_list, chunk_names = build_design_matrix(
        trial_info_list, T_trial, session_lengths, solutions=solutions
    )

    # Full-trial GLM
    full_res = run_permutation_glm(
        Y, X_list, boundaries,
        n_shifts=n_shifts, alpha=alpha, seed=seed, verbose=verbose,
    )

    if verbose:
        print(f"\nFull model R2: {full_res['r2_full']:.4f}")
        for k, name in enumerate(chunk_names):
            sig_str = "*" if full_res["significant"][k] else ""
            print(f"  {name:20s}  dR2={full_res['delta_r2_obs'][k]:.4f}  "
                  f"p={full_res['pvals'][k]:.4f} {sig_str}")

    # Time-resolved GLM
    if verbose:
        print(f"\nRunning time-resolved GLM...")

    tr_res = run_timeresolved(
        Y, trial_info_list, time_rel, T_trial,
        session_trial_counts,
        win_size=1.0, win_step=0.25,
        n_shifts=n_shifts, alpha=alpha, seed=seed,
        verbose=verbose,
    )

    return {
        "channel_id": channel_id,
        "full_glm": full_res,
        "timeresolved": tr_res,
        "chunk_names": chunk_names,
        "meta": {
            "sessions": sessions,
            "session_trial_counts": session_trial_counts,
            "T_trial": T_trial,
            "time_rel": time_rel,
            "n_shifts": n_shifts,
            "alpha": alpha,
        },
    }


# ====================================================================
# VISUALIZATION
# ====================================================================

def plot_results(all_results, save_dir=None):
    """
    Generate publication-quality summary figures.

    Parameters
    ----------
    all_results : dict mapping channel_id -> results dict
    save_dir : str or Path, optional
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Try to use Liberation Sans
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

    channels = list(all_results.keys())
    n_ch = len(channels)

    # --- Figure 1: Full-trial GLM summary (bar plot) ---
    fig, axes = plt.subplots(1, n_ch, figsize=(2.5 * n_ch + 0.5, 3.0),
                             sharey=True)
    if n_ch == 1:
        axes = [axes]

    predictor_names = ["solution", "lick_count", "lick_x_sol", "trial_num"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    for ci, ch in enumerate(channels):
        res = all_results[ch]["full_glm"]
        names = all_results[ch]["chunk_names"]
        # Only plot task-relevant predictors (exclude session)
        n_plot = min(4, len(names))
        dr2 = res["delta_r2_obs"][:n_plot]
        pv = res["pvals"][:n_plot]
        sig = res["significant"][:n_plot]

        x = np.arange(n_plot)
        bars = axes[ci].bar(x, dr2, color=colors[:n_plot], width=0.6,
                            edgecolor="black", linewidth=0.5)
        # Mark significance
        for xi in range(n_plot):
            if sig[xi]:
                axes[ci].text(xi, dr2[xi] + 0.002, "*",
                              ha="center", fontsize=12, fontweight="bold")

        axes[ci].set_xticks(x)
        axes[ci].set_xticklabels(names[:n_plot], rotation=45, ha="right",
                                  fontsize=7)
        axes[ci].set_title(ch, fontsize=10, fontweight="bold")
        axes[ci].spines["top"].set_visible(False)
        axes[ci].spines["right"].set_visible(False)

    axes[0].set_ylabel("$\\Delta R^2$")
    fig.suptitle("Full-trial GLM: unique variance explained",
                 fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "glm_full_trial_bar.png",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 2: Time-resolved GLM (line plots) ---
    fig, axes = plt.subplots(n_ch, 1, figsize=(5.5, 2.2 * n_ch),
                             sharex=True, squeeze=False)

    for ci, ch in enumerate(channels):
        tr = all_results[ch]["timeresolved"]
        ax = axes[ci, 0]
        n_plot = min(4, len(tr["chunk_names"]))
        for k in range(n_plot):
            label = tr["chunk_names"][k]
            ax.plot(tr["win_centers"], tr["delta_r2"][:, k],
                    color=colors[k], linewidth=1.2, label=label)
            # Shade significant windows
            sig_mask = tr["significant"][:, k]
            for wi in range(len(tr["win_centers"])):
                if sig_mask[wi]:
                    ax.axvspan(
                        tr["win_centers"][wi] - 0.125,
                        tr["win_centers"][wi] + 0.125,
                        alpha=0.15, color=colors[k], linewidth=0,
                    )
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_ylabel("$\\Delta R^2$")
        ax.set_title(ch, fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ci == 0:
            ax.legend(fontsize=7, frameon=False, ncol=2)

    axes[-1, 0].set_xlabel("Time from access onset (s)")
    fig.suptitle("Time-resolved GLM", fontsize=11, fontweight="bold",
                 y=1.01)
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "glm_timeresolved.png",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Figure 3: Null distribution with observed F ---
    fig, axes = plt.subplots(n_ch, 4, figsize=(7.2, 2.2 * n_ch),
                             squeeze=False)
    for ci, ch in enumerate(channels):
        res = all_results[ch]["full_glm"]
        names = all_results[ch]["chunk_names"]
        n_plot = min(4, len(names))
        for k in range(n_plot):
            ax = axes[ci, k]
            ax.hist(res["f_null"][:, k], bins=50, color="gray",
                    alpha=0.6, density=True, edgecolor="none")
            ax.axvline(res["f_obs"][k], color="red", linewidth=1.2)
            ax.set_title(f"{names[k]}\np={res['pvals'][k]:.4f}",
                         fontsize=7)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(labelsize=6)
            if k == 0:
                ax.set_ylabel(ch, fontsize=9, fontweight="bold")
        # Hide unused axes
        for k in range(n_plot, 4):
            axes[ci, k].set_visible(False)

    fig.suptitle("Null distributions (gray) vs observed F (red)",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    if save_dir:
        fig.savefig(Path(save_dir) / "glm_null_distributions.png",
                    dpi=300, bbox_inches="tight")
    plt.close(fig)

    if save_dir:
        print(f"\nFigures saved to {save_dir}")


# ====================================================================
# ENTRY POINT
# ====================================================================

def main(xlsx_path, save_dir=None, n_shifts=N_SHIFTS, alpha=ALPHA,
         seed=42, verbose=True):
    """
    Run the full analysis pipeline on the FP dataset.

    Parameters
    ----------
    xlsx_path : str or Path
    save_dir : str or Path, optional (defaults to same dir as xlsx)
    n_shifts : int
    alpha : float
    seed : int

    Returns
    -------
    all_results : dict mapping channel_id -> results
    """
    xlsx_path = Path(xlsx_path)
    if save_dir is None:
        save_dir = xlsx_path.parent
    save_dir = Path(save_dir)

    print("Loading data...")
    df = load_fp_data(xlsx_path)
    channels = sorted(df["channel_id"].unique())
    print(f"Channels: {channels}")
    print(f"Subjects: {sorted(df['subject'].unique())}")
    print(f"Solutions: {sorted(df['solution'].unique())}")

    all_results = {}
    for ch in channels:
        results = run_full_analysis(
            df, ch, n_shifts=n_shifts, alpha=alpha,
            seed=seed, verbose=verbose,
        )
        all_results[ch] = results

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Channel':<10} {'Predictor':<20} {'dR2':>8} {'p':>8} {'Sig':>5}")
    print("-" * 55)
    for ch in channels:
        res = all_results[ch]
        for k, name in enumerate(res["chunk_names"]):
            sig_str = "***" if res["full_glm"]["pvals"][k] < 0.001 else (
                "**" if res["full_glm"]["pvals"][k] < 0.01 else (
                    "*" if res["full_glm"]["pvals"][k] < 0.05 else ""))
            print(f"{ch:<10} {name:<20} "
                  f"{res['full_glm']['delta_r2_obs'][k]:>8.4f} "
                  f"{res['full_glm']['pvals'][k]:>8.4f} {sig_str:>5}")

    # Generate figures
    plot_results(all_results, save_dir=save_dir)

    # Save numerical results
    rows = []
    for ch in channels:
        res = all_results[ch]
        for k, name in enumerate(res["chunk_names"]):
            rows.append({
                "channel_id": ch,
                "predictor": name,
                "delta_r2": res["full_glm"]["delta_r2_obs"][k],
                "f_obs": res["full_glm"]["f_obs"][k],
                "p_value": res["full_glm"]["pvals"][k],
                "significant": res["full_glm"]["significant"][k],
                "r2_full_model": res["full_glm"]["r2_full"],
            })
    results_df = pd.DataFrame(rows)
    results_df.to_csv(save_dir / "glm_results_summary.csv", index=False)
    print(f"\nResults table saved to {save_dir / 'glm_results_summary.csv'}")

    return all_results


if __name__ == "__main__":
    import sys
    xlsx = sys.argv[1] if len(sys.argv) > 1 else (
        "streams_peth_multi_mix_pnq04_cbb08_cpb06.xlsx"
    )
    main(xlsx)
