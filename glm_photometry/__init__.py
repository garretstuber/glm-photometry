"""
glm-photometry: Circular-shift permutation GLM for fiber photometry data.

A framework for fitting generalized linear models to fiber photometry
peri-event time series using circular-shift permutation testing for
non-parametric significance evaluation.

Stuber Lab, University of Washington
"""

from .core import (
    ALPHA,
    N_SHIFTS,
    REFERENCE_SOLUTION,
    load_fp_data,
    get_trial_matrix,
    build_design_matrix,
    build_windowed_dm,
    circular_shift_1d,
    svd_setup,
    compute_fstats_1d,
    run_permutation_glm,
    run_timeresolved,
    prepare_channel_data,
    run_full_analysis,
    plot_results,
)

from .group_analysis import (
    load_h5_data,
    run_subject_glm,
)

from .per_solution import (
    build_per_solution_dm,
    build_per_solution_windowed_dm,
    run_timeresolved_per_sol,
    run_subject_glm_per_sol,
)

__version__ = "0.1.0"
__all__ = [
    # Constants
    "ALPHA", "N_SHIFTS", "REFERENCE_SOLUTION",
    # Data loading
    "load_fp_data", "load_h5_data",
    # Trial extraction
    "get_trial_matrix", "prepare_channel_data",
    # Design matrices
    "build_design_matrix", "build_windowed_dm",
    "build_per_solution_dm", "build_per_solution_windowed_dm",
    # GLM engine
    "circular_shift_1d", "svd_setup", "compute_fstats_1d",
    "run_permutation_glm", "run_timeresolved",
    # High-level runners
    "run_full_analysis", "run_subject_glm", "run_subject_glm_per_sol",
    "run_timeresolved_per_sol",
    # Visualization
    "plot_results",
]
