#!/usr/bin/env python3
"""
Analyze convergence of correlation metrics with increasing simulation cycles.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import glob
import re

def bootstrap_correlation(x, y, n_bootstrap=1000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for correlation metrics.

    Parameters:
    -----------
    x : array-like
        Independent variable (GroScore values)
    y : array-like
        Dependent variable (experimental pKd values)
    n_bootstrap : int
        Number of bootstrap resamples (default: 1000)
    confidence_level : float
        Confidence level for intervals (default: 0.95 for CI95)

    Returns:
    --------
    dict : Bootstrap statistics with CI95 for each metric
    """
    n = len(x)
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    # Initialize arrays to store bootstrap statistics
    pearson_bootstrap = np.zeros(n_bootstrap)
    spearman_bootstrap = np.zeros(n_bootstrap)
    r_squared_bootstrap = np.zeros(n_bootstrap)
    rmse_bootstrap = np.zeros(n_bootstrap)
    mae_bootstrap = np.zeros(n_bootstrap)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Perform bootstrap resampling
    i = 0
    attempts = 0
    max_attempts = n_bootstrap * 10  # Prevent infinite loop

    while i < n_bootstrap and attempts < max_attempts:
        attempts += 1

        # Resample with replacement
        indices = np.random.randint(0, n, size=n)
        x_boot = x[indices]
        y_boot = y[indices]

        # Skip if all x values are identical or all y values are identical
        if len(np.unique(x_boot)) < 2 or len(np.unique(y_boot)) < 2:
            continue

        try:
            # Fit linear regression on bootstrap sample
            slope_boot, intercept_boot, r_value_boot, _, _ = stats.linregress(x_boot, y_boot)
            y_pred_boot = slope_boot * x_boot + intercept_boot

            # Calculate metrics for this bootstrap sample
            pearson_bootstrap[i], _ = stats.pearsonr(y_pred_boot, y_boot)
            spearman_bootstrap[i], _ = stats.spearmanr(y_pred_boot, y_boot)
            r_squared_bootstrap[i] = r_value_boot**2
            rmse_bootstrap[i] = np.sqrt(np.mean((y_boot - y_pred_boot)**2))
            mae_bootstrap[i] = np.mean(np.abs(y_boot - y_pred_boot))

            i += 1  # Only increment if successful

        except (ValueError, RuntimeWarning):
            # Skip degenerate bootstrap samples
            continue

    if i < n_bootstrap:
        print(f"Warning: Only {i} valid bootstrap samples obtained out of {n_bootstrap} requested")

    # Use only valid bootstrap samples for percentile calculation
    pearson_bootstrap = pearson_bootstrap[:i]
    spearman_bootstrap = spearman_bootstrap[:i]
    r_squared_bootstrap = r_squared_bootstrap[:i]
    rmse_bootstrap = rmse_bootstrap[:i]
    mae_bootstrap = mae_bootstrap[:i]

    # Calculate confidence intervals using percentile method
    results = {
        'pearson': {
            'ci_lower': np.percentile(pearson_bootstrap, lower_percentile),
            'ci_upper': np.percentile(pearson_bootstrap, upper_percentile)
        },
        'spearman': {
            'ci_lower': np.percentile(spearman_bootstrap, lower_percentile),
            'ci_upper': np.percentile(spearman_bootstrap, upper_percentile)
        },
        'r_squared': {
            'ci_lower': np.percentile(r_squared_bootstrap, lower_percentile),
            'ci_upper': np.percentile(r_squared_bootstrap, upper_percentile)
        },
        'rmse': {
            'ci_lower': np.percentile(rmse_bootstrap, lower_percentile),
            'ci_upper': np.percentile(rmse_bootstrap, upper_percentile)
        },
        'mae': {
            'ci_lower': np.percentile(mae_bootstrap, lower_percentile),
            'ci_upper': np.percentile(mae_bootstrap, upper_percentile)
        }
    }

    return results

# Read experimental data
df = pd.read_csv('benchmark.csv')

# Find all cycle files and sort them
cycle_files = sorted(glob.glob('scores_avg_c*.gs'),
                     key=lambda x: int(re.search(r'c(\d+)', x).group(1)))

# Storage for convergence data
n_cycles = []
pearson_r_list = []
spearman_r_list = []
r_squared_list = []
rmse_list = []
mae_list = []
n_structures_list = []

# Storage for bootstrap confidence intervals
pearson_ci_lower = []
pearson_ci_upper = []
spearman_ci_lower = []
spearman_ci_upper = []
r_squared_ci_lower = []
r_squared_ci_upper = []
rmse_ci_lower = []
rmse_ci_upper = []
mae_ci_lower = []
mae_ci_upper = []

print("Convergence Analysis: GroScore vs Experimental pKd")
print("=" * 70)
print(f"\nExperimental pKd values from benchmark.csv:")
for _, row in df[['pdb_id', 'pkd']].head(10).iterrows():
    print(f"  {row['pdb_id']}: {row['pkd']:.2f}")
print()

for cycle_file in cycle_files:
    # Extract cycle number
    cycle_num = int(re.search(r'c(\d+)', cycle_file).group(1))

    # Read GroScore predictions for this cycle
    scores = {}
    with open(cycle_file, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                continue
            parts = line.strip().split()
            if len(parts) >= 2:
                structure_id = parts[0]
                score = float(parts[1])
                scores[structure_id] = score

    # Match structures and collect data
    groscore_values = []
    experimental_pkd = []

    for structure_id, score in scores.items():
        if structure_id in df['pdb_id'].values:
            pkd = df[df['pdb_id'] == structure_id]['pkd'].values[0]
            groscore_values.append(score)
            experimental_pkd.append(pkd)

    # Convert to arrays
    groscore_values = np.array(groscore_values)
    experimental_pkd = np.array(experimental_pkd)

    # Skip if not enough data (need at least 5 points for meaningful correlation)
    if len(groscore_values) < 5:
        continue

    # Convert GroScore to predicted pKd using linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(groscore_values, experimental_pkd)
    predicted_pkd = slope * groscore_values + intercept

    # Calculate correlation between predicted pKd and experimental pKd
    pearson_r, pearson_p = stats.pearsonr(predicted_pkd, experimental_pkd)
    spearman_r, spearman_p = stats.spearmanr(predicted_pkd, experimental_pkd)

    # Calculate error metrics
    r_squared = r_value**2
    rmse = np.sqrt(np.mean((experimental_pkd - predicted_pkd)**2))
    mae = np.mean(np.abs(experimental_pkd - predicted_pkd))

    # Store results
    n_cycles.append(cycle_num)
    pearson_r_list.append(pearson_r)
    spearman_r_list.append(spearman_r)
    r_squared_list.append(r_squared)
    rmse_list.append(rmse)
    mae_list.append(mae)
    n_structures_list.append(len(groscore_values))

    # Calculate bootstrap confidence intervals only if we have enough samples
    min_samples_for_bootstrap = 3
    if len(groscore_values) >= min_samples_for_bootstrap:
        print(f"\nCycle {cycle_num}: {len(groscore_values)} structures - calculating bootstrap CIs (n=1000)...")
        bootstrap_results = bootstrap_correlation(groscore_values, experimental_pkd, n_bootstrap=1000)

        # Store confidence intervals
        pearson_ci_lower.append(bootstrap_results['pearson']['ci_lower'])
        pearson_ci_upper.append(bootstrap_results['pearson']['ci_upper'])
        spearman_ci_lower.append(bootstrap_results['spearman']['ci_lower'])
        spearman_ci_upper.append(bootstrap_results['spearman']['ci_upper'])
        r_squared_ci_lower.append(bootstrap_results['r_squared']['ci_lower'])
        r_squared_ci_upper.append(bootstrap_results['r_squared']['ci_upper'])
        rmse_ci_lower.append(bootstrap_results['rmse']['ci_lower'])
        rmse_ci_upper.append(bootstrap_results['rmse']['ci_upper'])
        mae_ci_lower.append(bootstrap_results['mae']['ci_lower'])
        mae_ci_upper.append(bootstrap_results['mae']['ci_upper'])

        print(f"  Pearson r  = {pearson_r:7.4f}  CI95: [{bootstrap_results['pearson']['ci_lower']:7.4f}, {bootstrap_results['pearson']['ci_upper']:7.4f}]")
        print(f"  Spearman ρ = {spearman_r:7.4f}  CI95: [{bootstrap_results['spearman']['ci_lower']:7.4f}, {bootstrap_results['spearman']['ci_upper']:7.4f}]")
        print(f"  R²         = {r_squared:7.4f}  CI95: [{bootstrap_results['r_squared']['ci_lower']:7.4f}, {bootstrap_results['r_squared']['ci_upper']:7.4f}]")
        print(f"  RMSE       = {rmse:7.4f}  CI95: [{bootstrap_results['rmse']['ci_lower']:7.4f}, {bootstrap_results['rmse']['ci_upper']:7.4f}] pKd units")
        print(f"  MAE        = {mae:7.4f}  CI95: [{bootstrap_results['mae']['ci_lower']:7.4f}, {bootstrap_results['mae']['ci_upper']:7.4f}] pKd units")
    else:
        print(f"\nCycle {cycle_num}: {len(groscore_values)} structures")
        # Store NaN values for confidence intervals when sample size is too small
        pearson_ci_lower.append(np.nan)
        pearson_ci_upper.append(np.nan)
        spearman_ci_lower.append(np.nan)
        spearman_ci_upper.append(np.nan)
        r_squared_ci_lower.append(np.nan)
        r_squared_ci_upper.append(np.nan)
        rmse_ci_lower.append(np.nan)
        rmse_ci_upper.append(np.nan)
        mae_ci_lower.append(np.nan)
        mae_ci_upper.append(np.nan)

        print(f"  Pearson r  = {pearson_r:7.4f}")
        print(f"  Spearman ρ = {spearman_r:7.4f}")
        print(f"  R²         = {r_squared:7.4f}")
        print(f"  RMSE       = {rmse:7.4f} pKd units")
        print(f"  MAE        = {mae:7.4f} pKd units")
        print(f"  Note: Bootstrap CIs not calculated (n={len(groscore_values)} < {min_samples_for_bootstrap} samples)")

# Convert CI lists to numpy arrays
pearson_ci_lower = np.array(pearson_ci_lower)
pearson_ci_upper = np.array(pearson_ci_upper)
spearman_ci_lower = np.array(spearman_ci_lower)
spearman_ci_upper = np.array(spearman_ci_upper)
r_squared_ci_lower = np.array(r_squared_ci_lower)
r_squared_ci_upper = np.array(r_squared_ci_upper)
rmse_ci_lower = np.array(rmse_ci_lower)
rmse_ci_upper = np.array(rmse_ci_upper)
mae_ci_lower = np.array(mae_ci_lower)
mae_ci_upper = np.array(mae_ci_upper)

# Create convergence plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Check if any cycles have bootstrap CIs
has_any_ci = not np.all(np.isnan(pearson_ci_lower))
title_suffix = ' (with 95% Bootstrap CIs)' if has_any_ci else ''
fig.suptitle(f'Convergence of Correlation Metrics with Simulation Cycles{title_suffix}',
             fontsize=14, fontweight='bold')

# Plot 1: Pearson and Spearman correlation with confidence bands
ax1 = axes[0, 0]
ax1.plot(n_cycles, pearson_r_list, 'o-', linewidth=2, markersize=8, label='Pearson r', color='blue')
ax1.fill_between(n_cycles, pearson_ci_lower, pearson_ci_upper, alpha=0.2, color='blue')
ax1.plot(n_cycles, spearman_r_list, 's-', linewidth=2, markersize=8, label='Spearman ρ', color='green')
ax1.fill_between(n_cycles, spearman_ci_lower, spearman_ci_upper, alpha=0.2, color='green')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax1.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
title1 = 'Correlation Coefficients' + (' (with CI95)' if has_any_ci else '')
ax1.set_title(title1, fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(n_cycles)

# Plot 2: R² with confidence bands
ax2 = axes[0, 1]
ax2.plot(n_cycles, r_squared_list, 'o-', linewidth=2, markersize=8, color='red')
ax2.fill_between(n_cycles, r_squared_ci_lower, r_squared_ci_upper, alpha=0.2, color='red')
ax2.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
title2 = 'Coefficient of Determination' + (' (with CI95)' if has_any_ci else '')
ax2.set_title(title2, fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_cycles)
ax2.set_ylim(0, 1)

# Plot 3: RMSE and MAE with confidence bands
ax3 = axes[1, 0]
ax3.plot(n_cycles, rmse_list, 'o-', linewidth=2, markersize=8, label='RMSE', color='purple')
ax3.fill_between(n_cycles, rmse_ci_lower, rmse_ci_upper, alpha=0.2, color='purple')
ax3.plot(n_cycles, mae_list, 's-', linewidth=2, markersize=8, label='MAE', color='orange')
ax3.fill_between(n_cycles, mae_ci_lower, mae_ci_upper, alpha=0.2, color='orange')
ax3.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax3.set_ylabel('Error (pKd units)', fontsize=11, fontweight='bold')
title3 = 'Prediction Errors' + (' (with CI95)' if has_any_ci else '')
ax3.set_title(title3, fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(n_cycles)

# Plot 4: Number of structures
ax4 = axes[1, 1]
ax4.plot(n_cycles, n_structures_list, 'o-', linewidth=2, markersize=8, color='brown')
ax4.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Structures', fontsize=11, fontweight='bold')
ax4.set_title('Structures with Complete Data', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.set_xticks(n_cycles)
ax4.set_ylim(0, max(n_structures_list) + 1)

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=300, bbox_inches='tight')
print(f"\nConvergence plot saved to: convergence_plot.png")

# Summary statistics
print(f"\n{'Summary'}")
print("=" * 70)

# Check if we have bootstrap CIs for initial and final cycles
has_initial_ci = not np.isnan(pearson_ci_lower[0])
has_final_ci = not np.isnan(pearson_ci_lower[-1])

print(f"\nInitial (1 cycle):")
if has_initial_ci:
    print(f"  Pearson r  = {pearson_r_list[0]:.4f}  CI95: [{pearson_ci_lower[0]:.4f}, {pearson_ci_upper[0]:.4f}]")
    print(f"  R²         = {r_squared_list[0]:.4f}  CI95: [{r_squared_ci_lower[0]:.4f}, {r_squared_ci_upper[0]:.4f}]")
    print(f"  RMSE       = {rmse_list[0]:.4f}  CI95: [{rmse_ci_lower[0]:.4f}, {rmse_ci_upper[0]:.4f}]")
else:
    print(f"  Pearson r  = {pearson_r_list[0]:.4f}")
    print(f"  R²         = {r_squared_list[0]:.4f}")
    print(f"  RMSE       = {rmse_list[0]:.4f}")
    print(f"  (Bootstrap CIs not available - insufficient samples)")

print(f"\nFinal ({n_cycles[-1]} cycles):")
if has_final_ci:
    print(f"  Pearson r  = {pearson_r_list[-1]:.4f}  CI95: [{pearson_ci_lower[-1]:.4f}, {pearson_ci_upper[-1]:.4f}]")
    print(f"  R²         = {r_squared_list[-1]:.4f}  CI95: [{r_squared_ci_lower[-1]:.4f}, {r_squared_ci_upper[-1]:.4f}]")
    print(f"  RMSE       = {rmse_list[-1]:.4f}  CI95: [{rmse_ci_lower[-1]:.4f}, {rmse_ci_upper[-1]:.4f}]")
else:
    print(f"  Pearson r  = {pearson_r_list[-1]:.4f}")
    print(f"  R²         = {r_squared_list[-1]:.4f}")
    print(f"  RMSE       = {rmse_list[-1]:.4f}")
    print(f"  (Bootstrap CIs not available - insufficient samples)")

print(f"\nChanges from initial to final:")
print(f"  Change in r:     {pearson_r_list[-1] - pearson_r_list[0]:+.4f}")
print(f"  Change in R²:    {r_squared_list[-1] - r_squared_list[0]:+.4f}")
print(f"  Change in RMSE:  {rmse_list[-1] - rmse_list[0]:+.4f}")
