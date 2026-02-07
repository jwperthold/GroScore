#!/usr/bin/env python3
"""
Analyze correlation between GroScore predictions and experimental binding affinities.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd

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

print(f"\n{'Correlation Metrics'}")
print("-" * 70)
print(f"Pearson correlation:   r = {pearson_r:7.4f}  (p = {pearson_p:.4f})")
print(f"Spearman correlation:  ρ = {spearman_r:7.4f}  (p = {spearman_p:.4f})")
print(f"R²:                       {r_squared:7.4f}")
print(f"RMSE:                     {rmse:7.4f} pKd units")
print(f"MAE:                      {mae:7.4f} pKd units")

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
