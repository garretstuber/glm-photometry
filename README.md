# glm-photometry

Circular-shift permutation GLM for fiber photometry peri-event data.

A Python framework for fitting generalized linear models to fiber photometry time series using circular-shift permutation testing for non-parametric significance evaluation. Adapted from the calcium imaging GLM described in [Hjort et al.](https://doi.org/10.1016/j.crmeth.2024.100907) for single-channel fiber photometry recordings.

## Overview

This package provides tools to:

- Fit a GLM with task-relevant predictors (solution identity, lick count, trial number, etc.) to peri-event photometry signals
- Test predictor significance using circular-shift permutations that preserve temporal autocorrelation
- Compute delta-R2 (unique variance explained) for each predictor via leave-one-out F-tests
- Run time-resolved (sliding window) GLM to map encoding dynamics
- Decompose omnibus solution effects into individual solution contrasts (per-solution GLM)

## Installation

```bash
git clone https://github.com/garretstuber/glm-photometry.git
cd glm-photometry
pip install -e .
```

Or install dependencies directly:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import glm_photometry as glm

# Load data
df = glm.load_h5_data("path/to/streams_peth.h5")

# Run GLM for one subject
result = glm.run_subject_glm(df, subject="cba03", n_shifts=5000, alpha=0.01)

# Print results
for k, name in enumerate(result["chunk_names"]):
    sig = "*" if result["full_glm"]["significant"][k] else ""
    print(f"{name:20s}  dR2={result['full_glm']['delta_r2_obs'][k]:.4f}  "
          f"p={result['full_glm']['pvals'][k]:.4f} {sig}")
```

See the [notebook](notebooks/run_glm_photometry.ipynb) for a full walkthrough including group-level statistics and visualization.

## Package Structure

```
glm-photometry/
  glm_photometry/
    __init__.py          # Public API
    core.py              # GLM engine: design matrices, SVD, permutation testing
    group_analysis.py    # H5 data loader, per-subject GLM runner, group stats
    per_solution.py      # Per-solution contrast design matrices
  notebooks/
    run_glm_photometry.ipynb   # Interactive walkthrough
  scripts/
    run_batch.py               # Batch omnibus GLM
    run_batch_per_solution.py  # Batch per-solution GLM
    run_valence_timing.py      # Valence encoding and timing analysis
    run_decoding_and_splits.py # Single-trial decoding and early/late split
    ...
  pyproject.toml
  requirements.txt
  LICENSE
```

## Method Summary

### Design Matrix

The default design matrix contains five predictor chunks, each tested via a leave-one-out F-statistic:

| Chunk | Predictor | Columns | Description |
|-------|-----------|---------|-------------|
| 0 | Solution | K-1 | Dummy-coded solution identity (water = reference) |
| 1 | Lick count | 1 | Z-scored lick count per trial |
| 2 | Lick x Solution | K-1 | Interaction: lick count * solution dummies |
| 3 | Trial number | 1 | Linear drift within session (0 to 1) |
| 4 | Session | N-1 | Session indicator dummies (nuisance) |

The per-solution variant replaces the omnibus solution chunk with individual solution contrasts (e.g., sucrose vs. water, NaCl vs. water, dry vs. water), each as a separate 1-column chunk.

### Permutation Testing

Significance is evaluated by circularly shifting the signal within each session boundary to generate a null distribution of F-statistics. This preserves temporal autocorrelation while breaking the alignment between signal and predictors. P-values are computed as:

```
p = (sum(F_null >= F_obs) + 1) / (n_shifts + 1)
```

### SVD Acceleration

The residual sum of squares is computed efficiently via SVD decomposition: RSS = ||y||^2 - ||U'y||^2, avoiding explicit matrix inversion at each permutation.

## Data Format

Input data should be a tidy DataFrame (loaded from HDF5 or Excel) with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `subject` | str | Subject identifier |
| `channel_id` | str | Recording channel / population label |
| `blockname` | str | Session identifier |
| `solution` | str | Solution presented on each trial |
| `event_number_solution` | int | Trial number within solution |
| `time_rel` | float | Time relative to event onset (s) |
| `lick_count` | float | Lick count per trial |
| `delta_signal_poly_zscore_blsub` | float | Z-scored photometry signal |

## Dependencies

- Python >= 3.9
- NumPy >= 1.21
- Pandas >= 1.3
- SciPy >= 1.7
- h5py >= 3.0
- Matplotlib >= 3.4
- scikit-learn >= 1.0 (for decoding analysis)

## Citation

If you use this code, please cite:

Elerding et al. (in preparation). Functional characterization of VTA GABA neuron subtypes during a multi-solution taste access task.

Hjort et al. (2024). Bhatt-Lab circular-shift GLM for calcium imaging. *Cell Reports Methods*, 4(12), 100907.

## License

MIT License. See [LICENSE](LICENSE) for details.
