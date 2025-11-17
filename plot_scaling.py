#!/usr/bin/env python3
"""Plot scaling benchmark results"""

import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

# Find the most recent benchmark results directory
results_dirs = [d for d in os.listdir('.') if d.startswith('benchmark_scaling_')]
if not results_dirs:
    print("No benchmark results found!")
    sys.exit(1)

latest_dir = sorted(results_dirs)[-1]
csv_file = os.path.join(latest_dir, 'results.csv')

print(f"Reading results from: {csv_file}")

# Read data
df = pd.read_csv(csv_file)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('IPPL Independent Particles in Constant Magnetic Field\n100k particles, 10k timesteps, 64³ grid, 12th Gen Intel i7-1260P (16) @ 4.700GHz',
             fontsize=14, fontweight='bold')

# Plot 1: Runtime vs Threads
ax1 = axes[0, 0]
ax1.plot(df['Threads'], df['TaskParallel_Time'], 'o-', label='Task-Parallel', linewidth=2, markersize=8)
ax1.plot(df['Threads'], df['Master_Time'], 's-', label='Master', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Threads', fontsize=11)
ax1.set_ylabel('Runtime (seconds)', fontsize=11)
ax1.set_title('Runtime vs Thread Count', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xscale('log', base=2)
ax1.set_xticks(df['Threads'])
ax1.set_xticklabels(df['Threads'])

# Plot 2: Speedup (task-parallel vs master)
ax2 = axes[0, 1]
ax2.plot(df['Threads'], df['Speedup'], 'o-', color='green', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='red', linestyle='--', label='Equal performance', linewidth=1.5)
ax2.set_xlabel('Number of Threads', fontsize=11)
ax2.set_ylabel('Speedup (Task-Parallel / Master)', fontsize=11)
ax2.set_title('Task-Parallel Speedup vs Master', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xscale('log', base=2)
ax2.set_xticks(df['Threads'])
ax2.set_xticklabels(df['Threads'])

# Plot 3: Strong Scaling (speedup relative to 1 thread)
ax3 = axes[1, 0]
base_tp = df['TaskParallel_Time'].iloc[0]
base_master = df['Master_Time'].iloc[0]
tp_speedup = base_tp / df['TaskParallel_Time']
master_speedup = base_master / df['Master_Time']
ideal_speedup = df['Threads']

ax3.plot(df['Threads'], tp_speedup, 'o-', label='Task-Parallel', linewidth=2, markersize=8)
ax3.plot(df['Threads'], master_speedup, 's-', label='Master', linewidth=2, markersize=8)
ax3.plot(df['Threads'], ideal_speedup, 'k--', label='Ideal', linewidth=1.5, alpha=0.5)
ax3.set_xlabel('Number of Threads', fontsize=11)
ax3.set_ylabel('Speedup (relative to 1 thread)', fontsize=11)
ax3.set_title('Strong Scaling', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xscale('log', base=2)
ax3.set_yscale('log', base=2)
ax3.set_xticks(df['Threads'])
ax3.set_xticklabels(df['Threads'])

# Plot 4: Parallel Efficiency
ax4 = axes[1, 1]
tp_efficiency = 100 * tp_speedup / df['Threads']
master_efficiency = 100 * master_speedup / df['Threads']

ax4.plot(df['Threads'], tp_efficiency, 'o-', label='Task-Parallel', linewidth=2, markersize=8)
ax4.plot(df['Threads'], master_efficiency, 's-', label='Master', linewidth=2, markersize=8)
ax4.axhline(y=100, color='black', linestyle='--', label='Ideal (100%)', linewidth=1.5, alpha=0.5)
ax4.set_xlabel('Number of Threads', fontsize=11)
ax4.set_ylabel('Parallel Efficiency (%)', fontsize=11)
ax4.set_title('Parallel Efficiency', fontsize=12, fontweight='bold')
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xscale('log', base=2)
ax4.set_xticks(df['Threads'])
ax4.set_xticklabels(df['Threads'])
ax4.set_ylim([0, 110])

plt.tight_layout()

# Save figure
output_file = os.path.join(latest_dir, 'scaling_plots.png')
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Plot saved to: {output_file}")

# Also save as PDF for publication quality
output_pdf = os.path.join(latest_dir, 'scaling_plots.pdf')
plt.savefig(output_pdf, bbox_inches='tight')
print(f"PDF saved to: {output_pdf}")

plt.show()
