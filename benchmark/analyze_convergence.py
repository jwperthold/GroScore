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

    print(f"\nCycle {cycle_num}: {len(groscore_values)} structures")
    print(f"  Pearson r  = {pearson_r:7.4f}")
    print(f"  Spearman ρ = {spearman_r:7.4f}")
    print(f"  R²         = {r_squared:7.4f}")
    print(f"  RMSE       = {rmse:7.4f} pKd units")
    print(f"  MAE        = {mae:7.4f} pKd units")

# Create convergence plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Convergence of Correlation Metrics with Simulation Cycles',
             fontsize=14, fontweight='bold')

# Plot 1: Pearson and Spearman correlation
ax1 = axes[0, 0]
ax1.plot(n_cycles, pearson_r_list, 'o-', linewidth=2, markersize=8, label='Pearson r', color='blue')
ax1.plot(n_cycles, spearman_r_list, 's-', linewidth=2, markersize=8, label='Spearman ρ', color='green')
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax1.set_ylabel('Correlation Coefficient', fontsize=11, fontweight='bold')
ax1.set_title('Correlation Coefficients', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(n_cycles)

# Plot 2: R²
ax2 = axes[0, 1]
ax2.plot(n_cycles, r_squared_list, 'o-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax2.set_ylabel('R²', fontsize=11, fontweight='bold')
ax2.set_title('Coefficient of Determination', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xticks(n_cycles)
ax2.set_ylim(0, 1)

# Plot 3: RMSE and MAE
ax3 = axes[1, 0]
ax3.plot(n_cycles, rmse_list, 'o-', linewidth=2, markersize=8, label='RMSE', color='purple')
ax3.plot(n_cycles, mae_list, 's-', linewidth=2, markersize=8, label='MAE', color='orange')
ax3.set_xlabel('Number of Cycles', fontsize=11, fontweight='bold')
ax3.set_ylabel('Error (pKd units)', fontsize=11, fontweight='bold')
ax3.set_title('Prediction Errors', fontsize=12, fontweight='bold')
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
print(f"Initial (1 cycle):  r = {pearson_r_list[0]:.4f}, R² = {r_squared_list[0]:.4f}, RMSE = {rmse_list[0]:.4f}")
print(f"Final ({n_cycles[-1]} cycles): r = {pearson_r_list[-1]:.4f}, R² = {r_squared_list[-1]:.4f}, RMSE = {rmse_list[-1]:.4f}")
print(f"Change in r:     {pearson_r_list[-1] - pearson_r_list[0]:+.4f}")
print(f"Change in R²:    {r_squared_list[-1] - r_squared_list[0]:+.4f}")
print(f"Change in RMSE:  {rmse_list[-1] - rmse_list[0]:+.4f}")
