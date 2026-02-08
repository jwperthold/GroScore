#!/usr/bin/env python3
"""
Analyze correlation between GroScore predictions and experimental binding affinities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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

# Read GroScore predictions
scores = {}
with open('scores_avg.gs', 'r') as f:
    for line in f:
        if line.strip().startswith('#'):
            continue
        parts = line.strip().split()
        if len(parts) >= 2:
            structure_id = parts[0]
            score = float(parts[1])
            scores[structure_id] = score

# Read experimental data
df = pd.read_csv('benchmark.csv')

# Match structures and collect data
matched_structures = []
groscore_values = []
experimental_pkd = []

for structure_id, score in scores.items():
    if structure_id in df['pdb_id'].values:
        pkd = df[df['pdb_id'] == structure_id]['pkd'].values[0]
        matched_structures.append(structure_id)
        groscore_values.append(score)
        experimental_pkd.append(pkd)

# Convert to arrays
groscore_values = np.array(groscore_values)
experimental_pkd = np.array(experimental_pkd)

print(f"Correlation Analysis: GroScore vs Experimental pKd")
print("=" * 70)
print(f"\nMatched structures: {len(matched_structures)}")
print(f"Structures: {', '.join(matched_structures)}")

# First, convert GroScore to predicted pKd using linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(groscore_values, experimental_pkd)
predicted_pkd = slope * groscore_values + intercept

# Calculate correlations on pKd scale (predicted vs experimental)
pearson_r, pearson_p = stats.pearsonr(predicted_pkd, experimental_pkd)
spearman_r, spearman_p = stats.spearmanr(predicted_pkd, experimental_pkd)

# Calculate R² and error metrics
r_squared = r_value**2
rmse = np.sqrt(np.mean((experimental_pkd - predicted_pkd)**2))

# Calculate mean absolute error
mae = np.mean(np.abs(experimental_pkd - predicted_pkd))

# Calculate bootstrap confidence intervals only if we have enough samples
min_samples_for_bootstrap = 3
if len(groscore_values) >= min_samples_for_bootstrap:
    print(f"\nCalculating bootstrap confidence intervals (n=1000)...")
    bootstrap_results = bootstrap_correlation(groscore_values, experimental_pkd, n_bootstrap=1000)

    print(f"\n{'Correlation Metrics (with 95% Bootstrap Confidence Intervals)'}")
    print("-" * 70)
    print(f"Pearson correlation:   r = {pearson_r:7.4f}  CI95: [{bootstrap_results['pearson']['ci_lower']:7.4f}, {bootstrap_results['pearson']['ci_upper']:7.4f}]  (p = {pearson_p:.4f})")
    print(f"Spearman correlation:  ρ = {spearman_r:7.4f}  CI95: [{bootstrap_results['spearman']['ci_lower']:7.4f}, {bootstrap_results['spearman']['ci_upper']:7.4f}]  (p = {spearman_p:.4f})")
    print(f"R²:                       {r_squared:7.4f}  CI95: [{bootstrap_results['r_squared']['ci_lower']:7.4f}, {bootstrap_results['r_squared']['ci_upper']:7.4f}]")
    print(f"RMSE:                     {rmse:7.4f}  CI95: [{bootstrap_results['rmse']['ci_lower']:7.4f}, {bootstrap_results['rmse']['ci_upper']:7.4f}] pKd units")
    print(f"MAE:                      {mae:7.4f}  CI95: [{bootstrap_results['mae']['ci_lower']:7.4f}, {bootstrap_results['mae']['ci_upper']:7.4f}] pKd units")
else:
    print(f"\n{'Correlation Metrics'}")
    print("-" * 70)
    print(f"Pearson correlation:   r = {pearson_r:7.4f}  (p = {pearson_p:.4f})")
    print(f"Spearman correlation:  ρ = {spearman_r:7.4f}  (p = {spearman_p:.4f})")
    print(f"R²:                       {r_squared:7.4f}")
    print(f"RMSE:                     {rmse:7.4f} pKd units")
    print(f"MAE:                      {mae:7.4f} pKd units")
    print(f"\nNote: Bootstrap confidence intervals not calculated (n={len(groscore_values)} < {min_samples_for_bootstrap} samples)")

print(f"\n{'Linear Regression'}")
print("-" * 70)
print(f"pKd = {slope:.4f} × GroScore + {intercept:.4f}")
print(f"Standard error:           {std_err:.4f}")

print(f"\n{'Data Summary'}")
print("-" * 70)
print(f"GroScore range:        [{groscore_values.min():.1f}, {groscore_values.max():.1f}]")
print(f"Experimental pKd range: [{experimental_pkd.min():.2f}, {experimental_pkd.max():.2f}]")

print(f"\n{'Individual Results'}")
print("-" * 70)
print(f"{'Structure':<10} {'GroScore':>10} {'Exp pKd':>10} {'Pred pKd':>10} {'Error':>10}")
print("-" * 70)
for i, struct in enumerate(matched_structures):
    pred = predicted_pkd[i]
    error = experimental_pkd[i] - pred
    print(f"{struct:<10} {groscore_values[i]:>10.1f} {experimental_pkd[i]:>10.2f} {pred:>10.2f} {error:>10.2f}")

# Create scatter plot - Predicted vs Experimental pKd
plt.figure(figsize=(10, 8))
plt.scatter(experimental_pkd, predicted_pkd, s=100, alpha=0.7, edgecolors='black', linewidth=1.5)

# Add perfect prediction line (diagonal)
min_pkd = min(predicted_pkd.min(), experimental_pkd.min()) - 0.5
max_pkd = max(predicted_pkd.max(), experimental_pkd.max()) + 0.5
plt.plot([min_pkd, max_pkd], [min_pkd, max_pkd], 'r--', linewidth=2,
         label='Perfect prediction', alpha=0.7)

# Add labels for each point
for i, struct in enumerate(matched_structures):
    plt.annotate(struct, (experimental_pkd[i], predicted_pkd[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

plt.xlabel('Experimental pKd', fontsize=12, fontweight='bold')
plt.ylabel('Predicted pKd (from GroScore)', fontsize=12, fontweight='bold')
plt.title(f'GroScore Predictions vs Experimental Binding Affinity\n' +
          f'Pearson r = {pearson_r:.3f}, Spearman ρ = {spearman_r:.3f}, R² = {r_squared:.3f}, RMSE = {rmse:.2f}',
          fontsize=13, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(min_pkd, max_pkd)
plt.ylim(min_pkd, max_pkd)
plt.tight_layout()
plt.savefig('correlation_plot.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to: correlation_plot.png")
