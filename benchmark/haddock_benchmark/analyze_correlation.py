#!/usr/bin/env python3
"""
Analyze correlation between GroScore predictions and experimental binding affinities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

def bootstrap_correlation(x, y, n_bootstrap=50000, confidence_level=0.95):
    """
    Calculate bootstrap confidence intervals for correlation metrics.

    Parameters:
    -----------
    x : array-like
        Independent variable (GroScore values)
    y : array-like
        Dependent variable (experimental pKd values)
    n_bootstrap : int
        Number of bootstrap resamples (default: 50000)
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

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate all bootstrap indices at once (vectorized)
    indices = np.random.randint(0, n, size=(n_bootstrap, n))

    # Get all bootstrap samples at once
    x_boot = x[indices]  # shape: (n_bootstrap, n)
    y_boot = y[indices]  # shape: (n_bootstrap, n)

    # Identify degenerate samples (all x or y values identical)
    x_var = np.var(x_boot, axis=1)
    y_var = np.var(y_boot, axis=1)
    valid_mask = (x_var > 0) & (y_var > 0)

    if not np.any(valid_mask):
        raise ValueError("No valid bootstrap samples generated - all samples are degenerate")

    # Filter to valid samples only
    x_boot = x_boot[valid_mask]
    y_boot = y_boot[valid_mask]
    n_valid = x_boot.shape[0]

    if n_valid < n_bootstrap:
        print(f"Warning: Only {n_valid} valid bootstrap samples obtained out of {n_bootstrap} requested")

    # Vectorized linear regression calculations
    x_mean = np.mean(x_boot, axis=1, keepdims=True)
    y_mean = np.mean(y_boot, axis=1, keepdims=True)
    x_centered = x_boot - x_mean
    y_centered = y_boot - y_mean

    # Calculate slope and intercept for all bootstrap samples at once
    xy_cov = np.sum(x_centered * y_centered, axis=1)
    x_var = np.sum(x_centered**2, axis=1)
    slope_boot = xy_cov / x_var
    intercept_boot = y_mean.squeeze() - slope_boot * x_mean.squeeze()

    # Calculate predictions for all bootstrap samples
    y_pred_boot = slope_boot[:, np.newaxis] * x_boot + intercept_boot[:, np.newaxis]

    # Calculate R² (vectorized)
    ss_res = np.sum((y_boot - y_pred_boot)**2, axis=1)
    ss_tot = np.sum(y_centered**2, axis=1)
    r_squared_bootstrap = 1 - (ss_res / ss_tot)

    # Calculate RMSE and MAE (vectorized)
    residuals = y_boot - y_pred_boot
    rmse_bootstrap = np.sqrt(np.mean(residuals**2, axis=1))
    mae_bootstrap = np.mean(np.abs(residuals), axis=1)

    # Calculate Pearson correlation (vectorized)
    y_pred_centered = y_pred_boot - np.mean(y_pred_boot, axis=1, keepdims=True)
    numerator = np.sum(y_pred_centered * y_centered, axis=1)
    denominator = np.sqrt(np.sum(y_pred_centered**2, axis=1) * np.sum(y_centered**2, axis=1))
    pearson_bootstrap = numerator / denominator

    # Calculate Spearman correlation using vectorized ranking
    # Rank each bootstrap sample independently
    from scipy.stats import rankdata

    # Rank predictions and actual values for all bootstrap samples
    # Using 'average' method to handle ties
    ranked_pred = np.array([rankdata(y_pred_boot[i], method='average') for i in range(n_valid)])
    ranked_actual = np.array([rankdata(y_boot[i], method='average') for i in range(n_valid)])

    # Calculate Pearson correlation on ranks (which is Spearman correlation)
    pred_mean = np.mean(ranked_pred, axis=1, keepdims=True)
    actual_mean = np.mean(ranked_actual, axis=1, keepdims=True)
    pred_centered = ranked_pred - pred_mean
    actual_centered = ranked_actual - actual_mean

    numerator = np.sum(pred_centered * actual_centered, axis=1)
    denominator = np.sqrt(np.sum(pred_centered**2, axis=1) * np.sum(actual_centered**2, axis=1))

    # Handle potential division by zero (constant ranks)
    with np.errstate(divide='ignore', invalid='ignore'):
        spearman_bootstrap = numerator / denominator
        spearman_bootstrap = np.nan_to_num(spearman_bootstrap, nan=0.0)

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
            try:
                score = float(parts[1])
                if not np.isnan(score):
                    scores[structure_id] = score
            except ValueError:
                pass

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
    print(f"\nCalculating bootstrap confidence intervals (n=50000)...")
    bootstrap_results = bootstrap_correlation(groscore_values, experimental_pkd, n_bootstrap=50000)

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
plt.scatter(experimental_pkd, predicted_pkd, s=50, alpha=0.7, edgecolors='black', linewidth=1.0)

# Add perfect prediction line (diagonal)
min_pkd = min(predicted_pkd.min(), experimental_pkd.min()) - 0.5
max_pkd = max(predicted_pkd.max(), experimental_pkd.max()) + 0.5
plt.plot([min_pkd, max_pkd], [min_pkd, max_pkd], 'k--', linewidth=1,
         label='Identity', alpha=0.7)
plt.plot([min_pkd-1.0, max_pkd-1.0], [min_pkd, max_pkd], 'k:', linewidth=1,
         label='Identity + 1.0', alpha=0.2)
plt.plot([min_pkd+1.0, max_pkd+1.0], [min_pkd, max_pkd], 'k:', linewidth=1,
         label='Identity - 1.0', alpha=0.2)

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
