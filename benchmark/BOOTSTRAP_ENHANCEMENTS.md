# Bootstrap Confidence Interval Enhancements

## Overview
Added 95% bootstrap confidence intervals (CI95) to both correlation and convergence analysis scripts using 50,000 bootstrap replicas for robust uncertainty quantification. **Bootstrap calculations are automatically skipped when sample size is insufficient (n < 3).**

## Changes to analyze_correlation.py

### Bootstrap Function
- **Function**: `bootstrap_correlation(x, y, n_bootstrap=50000, confidence_level=0.95)`
- **Method**: Percentile method (2.5th and 97.5th percentiles)
- **Replicas**: 50,000 bootstrap resamples with replacement
- **Random seed**: 42 (for reproducibility)
- **Minimum samples**: 3 (bootstrap skipped if n < 3)

### Metrics with CI95
All correlation metrics report confidence intervals when sufficient data is available:
- Pearson correlation (r)
- Spearman correlation (ρ)
- R² (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)

### Output Format

**With sufficient samples (n ≥ 3):**
```
Calculating bootstrap confidence intervals (n=50000)...

Correlation Metrics (with 95% Bootstrap Confidence Intervals)
----------------------------------------------------------------------
Pearson correlation:   r =  0.8234  CI95: [ 0.7123,  0.9012]  (p = 0.0001)
Spearman correlation:  ρ =  0.7891  CI95: [ 0.6745,  0.8834]  (p = 0.0003)
R²:                       0.6780  CI95: [ 0.5071,  0.8120]
RMSE:                     0.8456  CI95: [ 0.6234,  1.0123] pKd units
MAE:                      0.6789  CI95: [ 0.4912,  0.8234] pKd units
```

**With insufficient samples (n < 3):**
```
Correlation Metrics
----------------------------------------------------------------------
Pearson correlation:   r =  0.8234  (p = 0.0001)
Spearman correlation:  ρ =  0.7891  (p = 0.0003)
R²:                       0.6780
RMSE:                     0.8456 pKd units
MAE:                      0.6789 pKd units

Note: Bootstrap confidence intervals not calculated (n=2 < 3 samples)
```

## Changes to analyze_convergence.py

### Bootstrap Function
Same implementation as analyze_correlation.py - calculates CI95 for each convergence cycle with sufficient samples.

### Enhanced Visualization
Plots dynamically adapt based on whether any cycles have bootstrap CIs:
1. **Correlation Coefficients Plot**: Shaded bands for Pearson r and Spearman ρ (when available)
2. **R² Plot**: Shaded band showing uncertainty in coefficient of determination (when available)
3. **Prediction Errors Plot**: Shaded bands for both RMSE and MAE (when available)
4. **Structures Plot**: Unchanged (no uncertainty)

Title automatically adjusts: "(with 95% Bootstrap CIs)" only shown if at least one cycle has CIs.

### Cycle-by-Cycle Output

**With sufficient samples (n ≥ 3):**
```
Cycle 5: 42 structures - calculating bootstrap CIs (n=50000)...
  Pearson r  =  0.8234  CI95: [ 0.7123,  0.9012]
  Spearman ρ =  0.7891  CI95: [ 0.6745,  0.8834]
  R²         =  0.6780  CI95: [ 0.5071,  0.8120]
  RMSE       =  0.8456  CI95: [ 0.6234,  1.0123] pKd units
  MAE        =  0.6789  CI95: [ 0.4912,  0.8234] pKd units
```

**With insufficient samples (n < 3):**
```
Cycle 1: 2 structures
  Pearson r  =  0.8234
  Spearman ρ =  0.7891
  R²         =  0.6780
  RMSE       =  0.8456 pKd units
  MAE        =  0.6789 pKd units
  Note: Bootstrap CIs not calculated (n=2 < 3 samples)
```

### Enhanced Summary

**When both initial and final cycles have CIs:**
```
Summary
======================================================================

Initial (1 cycle):
  Pearson r  = 0.7532  CI95: [0.6234, 0.8543]
  R²         = 0.5673  CI95: [0.3887, 0.7298]
  RMSE       = 0.9823  CI95: [0.7456, 1.2345]

Final (10 cycles):
  Pearson r  = 0.8456  CI95: [0.7532, 0.9123]
  R²         = 0.7150  CI95: [0.5677, 0.8323]
  RMSE       = 0.7234  CI95: [0.5678, 0.9012]

Changes from initial to final:
  Change in r:     +0.0924
  Change in R²:    +0.1477
  Change in RMSE:  -0.2589
```

**When initial cycle lacks CIs:**
```
Summary
======================================================================

Initial (1 cycle):
  Pearson r  = 0.7532
  R²         = 0.5673
  RMSE       = 0.9823
  (Bootstrap CIs not available - insufficient samples)

Final (10 cycles):
  Pearson r  = 0.8456  CI95: [0.7532, 0.9123]
  R²         = 0.7150  CI95: [0.5677, 0.8323]
  RMSE       = 0.7234  CI95: [0.5678, 0.9012]

Changes from initial to final:
  Change in r:     +0.0924
  Change in R²:    +0.1477
  Change in RMSE:  -0.2589
```

## Statistical Methodology

### Bootstrap Resampling
- **Approach**: Non-parametric bootstrap with replacement
- **Sample size**: n (original dataset size) for each bootstrap sample
- **Replicas**: 50,000 iterations
- **Random seed**: Fixed at 42 for reproducibility
- **Minimum sample threshold**: 3 structures (configurable via `min_samples_for_bootstrap`)

### Confidence Interval Calculation
- **Method**: Percentile method
- **Confidence level**: 95% (α = 0.05)
- **Lower bound**: 2.5th percentile of bootstrap distribution
- **Upper bound**: 97.5th percentile of bootstrap distribution

### Bootstrap Process
For each bootstrap replica (when n ≥ 3):
1. Resample (x, y) pairs with replacement
2. Fit linear regression on bootstrap sample
3. Calculate predicted pKd values
4. Compute all correlation metrics (Pearson, Spearman, R², RMSE, MAE)
5. Store results

After all replicas:
- Sort bootstrap distributions for each metric
- Extract 2.5th and 97.5th percentiles as CI95 bounds

### Handling Insufficient Samples
When n < 3:
- Bootstrap calculation skipped entirely
- Point estimates still calculated and reported
- NaN values stored for CI bounds (in convergence analysis)
- Clear message indicates why CIs are not available
- Plots omit confidence bands for cycles with NaN CIs (matplotlib handles this gracefully)

## Performance Considerations

### Computational Cost
- **50,000 replicas with n ≥ 3**: ~15-30 seconds per dataset (depends on sample size)
- **n < 3**: Instant (bootstrap skipped)
- **analyze_correlation.py**: Single bootstrap run or skip
- **analyze_convergence.py**: One bootstrap run or skip per cycle file

### Memory Usage
- Negligible: Bootstrap statistics stored as simple arrays
- No raw bootstrap samples retained after CI calculation
- NaN values (8 bytes) used for cycles without CIs

## Benefits

1. **Robust Uncertainty Quantification**: Non-parametric approach handles non-normal distributions
2. **Visual Interpretation**: Shaded bands in plots make uncertainty immediately apparent
3. **Statistical Rigor**: 50,000 replicas provide stable, reliable confidence intervals
4. **Reproducibility**: Fixed random seed ensures consistent results across runs
5. **Comprehensive**: CI95 calculated for all five correlation metrics when possible
6. **Automatic Handling**: Scripts intelligently skip bootstrap for small samples
7. **Clear Communication**: Users informed when/why CIs are not calculated

## Why Minimum 3 Samples?

Bootstrap confidence intervals require resampling with replacement. With only 1 or 2 samples:
- **n=1**: All bootstrap samples identical (no resampling variation)
- **n=2**: Very limited bootstrap distribution (only 3 possible unique samples: {A,A}, {B,B}, {A,B})
- **n≥3**: Sufficient variation for meaningful bootstrap distributions

The threshold of 3 is conservative - some practitioners use 5-10. This can be adjusted via the `min_samples_for_bootstrap` variable in each script.

## Usage Notes

- Scripts automatically calculate bootstrap CIs when sample size permits
- No additional command-line arguments required
- Execution time increases by ~15-30 seconds per dataset with sufficient samples
- Results remain reproducible with fixed random seed
- Small samples still get point estimates reported
