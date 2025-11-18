#!/usr/bin/env python3

"""
CUDA Branch Comparison Plotting Script
Compares task-parallel vs master branch on CUDA across different particle counts
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

def load_results(csv_path):
    """Load benchmark results from CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except FileNotFoundError:
        print(f"Error: Could not find {csv_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

def plot_cuda_comparison(df, output_dir):
    """Create comprehensive CUDA branch comparison plots."""

    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('IndependentParticlesTest Performance Analysis on NVIDIA T550 Laptop GPU',
                 fontsize=16, fontweight='bold')

    # Plot 1: Execution Time Comparison
    ax1 = axes[0, 0]
    x_pos = np.arange(len(df['Label']))
    width = 0.35

    bars1 = ax1.bar(x_pos - width/2, df['TaskParallel_Time_s'], width,
                    label='Task-Parallel', color='#2E86AB', alpha=0.8)
    bars2 = ax1.bar(x_pos + width/2, df['Master_Time_s'], width,
                    label='Master', color='#A23B72', alpha=0.8)

    ax1.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('Execution Time: Task-Parallel vs Master', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(df['Label'], rotation=45)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    # Plot 2: Speedup across Particle Counts
    ax2 = axes[0, 1]
    speedup = df['Speedup']

    colors = ['#06A77D' if s >= 1 else '#D62246' for s in speedup]
    bars = ax2.bar(df['Label'], speedup, color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7, label='No speedup (1.0x)')

    ax2.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup (Master / Task-Parallel)', fontsize=11, fontweight='bold')
    ax2.set_title('Task-Parallel Speedup vs Master', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, s in zip(bars, speedup):
        height = bar.get_height()
        ax2.annotate(f'{s:.2f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Plot 3: Throughput Comparison
    ax3 = axes[1, 0]

    task_parallel_throughput_millions = df['TaskParallel_Throughput'] / 1e6
    master_throughput_millions = df['Master_Throughput'] / 1e6

    ax3.plot(df['Label'], task_parallel_throughput_millions, marker='o',
             linewidth=2, markersize=8, label='Task-Parallel', color='#2E86AB')
    ax3.plot(df['Label'], master_throughput_millions, marker='s',
             linewidth=2, markersize=8, label='Master', color='#A23B72')

    ax3.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Throughput (Million Particle-Updates/s)', fontsize=11, fontweight='bold')
    ax3.set_title('Throughput Comparison', fontsize=12, fontweight='bold')
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Relative Performance (Percentage Improvement)
    ax4 = axes[1, 1]

    # Calculate percentage improvement: (Master - TaskParallel) / Master * 100
    # Positive = task-parallel is faster, Negative = master is faster
    pct_improvement = ((df['Master_Time_s'] - df['TaskParallel_Time_s']) / df['Master_Time_s']) * 100

    colors = ['#06A77D' if p >= 0 else '#D62246' for p in pct_improvement]
    bars = ax4.bar(df['Label'], pct_improvement, color=colors, alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)

    ax4.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Performance Improvement (%)', fontsize=11, fontweight='bold')
    ax4.set_title('Task-Parallel Performance Improvement over Master', fontsize=12, fontweight='bold')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, pct in zip(bars, pct_improvement):
        height = bar.get_height()
        label_y = height + (1 if height >= 0 else -3)
        ax4.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -3),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'cuda_comparison_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Create additional detailed comparison plot
    fig2, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(12, 10))
    fig2.suptitle('IndependentParticlesTest Performance Analysis on NVIDIA T550 Laptop GPU', fontsize=14, fontweight='bold')

    # Top plot: Time vs Particle Count for both branches
    particles_millions = df['Particles'] / 1e6

    ax_top.scatter(particles_millions, df['TaskParallel_Time_s'], s=100, alpha=0.6,
                   color='#2E86AB', label='Task-Parallel')
    ax_top.plot(particles_millions, df['TaskParallel_Time_s'], linewidth=2,
                alpha=0.8, color='#2E86AB')

    ax_top.scatter(particles_millions, df['Master_Time_s'], s=100, alpha=0.6,
                   color='#A23B72', label='Master')
    ax_top.plot(particles_millions, df['Master_Time_s'], linewidth=2,
                alpha=0.8, color='#A23B72')

    # Fit linear curves
    z_tp = np.polyfit(particles_millions, df['TaskParallel_Time_s'], 1)
    z_m = np.polyfit(particles_millions, df['Master_Time_s'], 1)
    p_tp = np.poly1d(z_tp)
    p_m = np.poly1d(z_m)

    x_line = np.linspace(particles_millions.min(), particles_millions.max(), 100)
    ax_top.plot(x_line, p_tp(x_line), "--", alpha=0.5, color='#2E86AB',
                label=f'Task-Parallel fit: y={z_tp[0]:.3f}x+{z_tp[1]:.2f}')
    ax_top.plot(x_line, p_m(x_line), "--", alpha=0.5, color='#A23B72',
                label=f'Master fit: y={z_m[0]:.3f}x+{z_m[1]:.2f}')

    ax_top.set_xlabel('Particle Count (Millions)', fontsize=12, fontweight='bold')
    ax_top.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax_top.set_title('Execution Time vs Particle Count', fontsize=13, fontweight='bold')
    ax_top.legend(loc='best')
    ax_top.grid(True, alpha=0.3)

    # Bottom plot: Speedup trend
    ax_bottom.plot(particles_millions, df['Speedup'], marker='D', linewidth=2.5,
                   markersize=10, color='#F18F01')
    ax_bottom.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7,
                     label='No speedup (1.0x)')

    # Shade regions
    ax_bottom.fill_between(particles_millions, 1.0, df['Speedup'].max() + 0.1,
                           where=(df['Speedup'] >= 1.0), alpha=0.2, color='green',
                           label='Task-parallel faster')
    ax_bottom.fill_between(particles_millions, df['Speedup'].min() - 0.1, 1.0,
                           where=(df['Speedup'] < 1.0), alpha=0.2, color='red',
                           label='Master faster')

    # Add labels for each point
    for i, (pcount, speedup, label) in enumerate(zip(particles_millions, df['Speedup'], df['Label'])):
        ax_bottom.annotate(f'{label}\n{speedup:.2f}x',
                          (pcount, speedup),
                          textcoords="offset points",
                          xytext=(0, 10),
                          ha='center',
                          fontsize=9,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax_bottom.set_xlabel('Particle Count (Millions)', fontsize=12, fontweight='bold')
    ax_bottom.set_ylabel('Speedup (Master / Task-Parallel)', fontsize=12, fontweight='bold')
    ax_bottom.set_title('Speedup Trend with Particle Count', fontsize=13, fontweight='bold')
    ax_bottom.legend(loc='best')
    ax_bottom.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path2 = os.path.join(output_dir, 'cuda_comparison_detailed.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path2}")

    plt.show()

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*70)
    print("CUDA BRANCH COMPARISON SUMMARY")
    print("="*70)

    print(f"\nParticle counts tested: {len(df)}")
    print(f"  Range: {df['Particles'].min():,} - {df['Particles'].max():,} particles")

    # Time statistics
    print(f"\nTask-Parallel Time Range: {df['TaskParallel_Time_s'].min():.2f}s - {df['TaskParallel_Time_s'].max():.2f}s")
    print(f"Master Time Range:        {df['Master_Time_s'].min():.2f}s - {df['Master_Time_s'].max():.2f}s")

    # Speedup statistics
    avg_speedup = df['Speedup'].mean()
    max_speedup = df['Speedup'].max()
    min_speedup = df['Speedup'].min()

    print(f"\nSpeedup Statistics (Master / Task-Parallel):")
    print(f"  Average: {avg_speedup:.3f}x")
    print(f"  Maximum: {max_speedup:.3f}x ({df.loc[df['Speedup'].idxmax(), 'Label']})")
    print(f"  Minimum: {min_speedup:.3f}x ({df.loc[df['Speedup'].idxmin(), 'Label']})")

    if avg_speedup > 1.0:
        avg_improvement = (avg_speedup - 1.0) * 100
        print(f"\n  → Task-parallel is on average {avg_improvement:.1f}% faster")
    elif avg_speedup < 1.0:
        avg_slowdown = (1.0 - avg_speedup) * 100
        print(f"\n  → Task-parallel is on average {avg_slowdown:.1f}% slower")
    else:
        print(f"\n  → Both branches have identical performance")

    # Throughput statistics
    avg_tp_throughput = df['TaskParallel_Throughput'].mean() / 1e6
    avg_m_throughput = df['Master_Throughput'].mean() / 1e6

    print(f"\nAverage Throughput (Million particle-updates/s):")
    print(f"  Task-Parallel: {avg_tp_throughput:.2f}M")
    print(f"  Master:        {avg_m_throughput:.2f}M")

    # Scaling analysis
    print(f"\nScaling Analysis (smallest to largest):")

    smallest_idx = 0
    largest_idx = len(df) - 1

    tp_time_ratio = df['TaskParallel_Time_s'].iloc[largest_idx] / df['TaskParallel_Time_s'].iloc[smallest_idx]
    m_time_ratio = df['Master_Time_s'].iloc[largest_idx] / df['Master_Time_s'].iloc[smallest_idx]
    particle_ratio = df['Particles'].iloc[largest_idx] / df['Particles'].iloc[smallest_idx]

    tp_efficiency = (particle_ratio / tp_time_ratio) * 100
    m_efficiency = (particle_ratio / m_time_ratio) * 100

    print(f"  Particles increased:         {particle_ratio:.1f}x")
    print(f"  Task-Parallel time increased: {tp_time_ratio:.2f}x (efficiency: {tp_efficiency:.1f}%)")
    print(f"  Master time increased:        {m_time_ratio:.2f}x (efficiency: {m_efficiency:.1f}%)")

    print("="*70 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_cuda_comparison.py <results_directory>")
        print("\nExample:")
        print("  python plot_cuda_comparison.py benchmark_cuda_comparison_20251117_120000/")
        sys.exit(1)

    results_dir = sys.argv[1]
    csv_path = os.path.join(results_dir, 'results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Could not find results.csv in {results_dir}")
        print(f"Expected path: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")
    df = load_results(csv_path)

    print(f"Found {len(df)} benchmark results")
    print("\nData preview:")
    print(df[['Label', 'Particles', 'TaskParallel_Time_s', 'Master_Time_s', 'Speedup']].to_string(index=False))

    print_summary(df)

    print("\nGenerating plots...")
    plot_cuda_comparison(df, results_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
