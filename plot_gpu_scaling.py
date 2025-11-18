#!/usr/bin/env python3

"""
GPU Scaling Benchmark Plotting Script
Visualizes performance scaling with increasing problem sizes
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

def plot_gpu_scaling(df, output_dir):
    """Create comprehensive GPU scaling plots."""

    # Set up the plot style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Extract timesteps from throughput calculation
    # Throughput = Total_Updates / Time = (Particles * Timesteps) / Time
    # So Timesteps = (Throughput * Time) / Particles
    TIMESTEPS = int((df['Throughput'].iloc[0] * df['Time_s'].iloc[0]) / df['Particles'].iloc[0])

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    grid_size = f"{df['Grid_NX'].iloc[0]}×{df['Grid_NY'].iloc[0]}×{df['Grid_NZ'].iloc[0]}"
    fig.suptitle(f'IPPL GPU Performance Scaling with Particle Count (Grid: {grid_size})',
                 fontsize=16, fontweight='bold')

    # Plot 1: Execution Time vs Particle Count
    ax1 = axes[0, 0]
    ax1.plot(df['Label'], df['Time_s'], marker='o', linewidth=2, markersize=8, color='#2E86AB')
    ax1.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Execution Time (seconds)', fontsize=11, fontweight='bold')
    ax1.set_title('GPU Execution Time vs Particle Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add value labels on points
    for i, (label, time) in enumerate(zip(df['Label'], df['Time_s'])):
        ax1.annotate(f'{time:.2f}s',
                    (i, time),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=9)

    # Plot 2: Throughput (Particle Updates per Second)
    ax2 = axes[0, 1]
    throughput_millions = df['Throughput'] / 1e6
    ax2.plot(df['Label'], throughput_millions, marker='s', linewidth=2, markersize=8, color='#A23B72')
    ax2.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Throughput (Million Particle-Updates/s)', fontsize=11, fontweight='bold')
    ax2.set_title('GPU Throughput vs Particle Count', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, (label, tput) in enumerate(zip(df['Label'], throughput_millions)):
        ax2.annotate(f'{tput:.2f}M',
                    (i, tput),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=9)

    # Plot 3: Scaling Efficiency
    ax3 = axes[1, 0]
    # Calculate ideal scaling (linear) vs actual
    base_time = df['Time_s'].iloc[0]
    base_particles = df['Particles'].iloc[0]

    ideal_time = base_time * (df['Particles'] / base_particles)
    actual_time = df['Time_s']
    efficiency = (ideal_time / actual_time) * 100

    ax3.plot(df['Label'], efficiency, marker='^', linewidth=2, markersize=8, color='#F18F01', label='Actual Efficiency')
    ax3.axhline(y=100, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (100%)')
    ax3.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Scaling Efficiency (%)', fontsize=11, fontweight='bold')
    ax3.set_title('GPU Scaling Efficiency (vs Ideal Linear Scaling)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)
    ax3.legend(loc='best')
    ax3.set_ylim(bottom=0)

    # Add value labels
    for i, (label, eff) in enumerate(zip(df['Label'], efficiency)):
        ax3.annotate(f'{eff:.1f}%',
                    (i, eff),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=9)

    # Plot 4: Time per Particle
    ax4 = axes[1, 1]

    # Calculate time per million particles per timestep
    time_per_particle = (df['Time_s'] * 1e6) / (df['Particles'] * TIMESTEPS)

    ax4.plot(df['Label'], time_per_particle, marker='D', linewidth=2, markersize=8, color='#D62246')
    ax4.set_xlabel('Particle Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Time per Million Particles per Timestep (µs)', fontsize=11, fontweight='bold')
    ax4.set_title('GPU Performance: Time per Particle', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(axis='x', rotation=45)

    # Add value labels
    for i, (label, tpp) in enumerate(zip(df['Label'], time_per_particle)):
        ax4.annotate(f'{tpp:.2f}µs',
                    (i, tpp),
                    textcoords="offset points",
                    xytext=(0,10),
                    ha='center',
                    fontsize=9)

    plt.tight_layout()

    # Save the figure
    output_path = os.path.join(output_dir, 'gpu_scaling_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")

    # Create additional plot: Time vs Particle Count
    fig2, ax = plt.subplots(figsize=(10, 6))

    particles_millions = df['Particles'] / 1e6
    ax.scatter(particles_millions, df['Time_s'], s=100, alpha=0.6, color='#2E86AB')
    ax.plot(particles_millions, df['Time_s'], linewidth=2, alpha=0.8, color='#2E86AB')

    # Add labels for each point
    for i, (pcount, time, label) in enumerate(zip(particles_millions, df['Time_s'], df['Label'])):
        ax.annotate(label,
                   (pcount, time),
                   textcoords="offset points",
                   xytext=(10,5),
                   ha='left',
                   fontsize=10,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))

    ax.set_xlabel('Particle Count (Millions)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Execution Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title(f'GPU Performance: Execution Time vs Particle Count (Grid: {grid_size})',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Fit a curve to show scaling behavior
    z = np.polyfit(particles_millions, df['Time_s'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(particles_millions.min(), particles_millions.max(), 100)
    ax.plot(x_line, p(x_line), "--", alpha=0.5, color='red',
            label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.2f}')
    ax.legend(loc='best')

    plt.tight_layout()

    output_path2 = os.path.join(output_dir, 'gpu_time_vs_particles.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path2}")

    plt.show()

def print_summary(df):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("GPU PARTICLE SCALING BENCHMARK SUMMARY")
    print("="*60)
    grid_size = f"{df['Grid_NX'].iloc[0]}×{df['Grid_NY'].iloc[0]}×{df['Grid_NZ'].iloc[0]}"
    print(f"\nFixed grid size: {grid_size}")
    print(f"Particle counts tested: {len(df)}")
    print(f"  Range: {df['Particles'].min():,} - {df['Particles'].max():,} particles")
    print(f"\nTime range: {df['Time_s'].min():.2f}s - {df['Time_s'].max():.2f}s")
    print(f"Time scaling: {df['Time_s'].max() / df['Time_s'].min():.2f}x slower for largest problem")

    # Calculate throughput statistics
    avg_throughput = df['Throughput'].mean() / 1e6
    max_throughput = df['Throughput'].max() / 1e6
    min_throughput = df['Throughput'].min() / 1e6

    print(f"\nThroughput statistics (Million particle-updates/s):")
    print(f"  Average: {avg_throughput:.2f}M")
    print(f"  Maximum: {max_throughput:.2f}M ({df.loc[df['Throughput'].idxmax(), 'Label']})")
    print(f"  Minimum: {min_throughput:.2f}M ({df.loc[df['Throughput'].idxmin(), 'Label']})")

    # Calculate scaling efficiency for largest problem
    base_time = df['Time_s'].iloc[0]
    base_particles = df['Particles'].iloc[0]
    largest_time = df['Time_s'].iloc[-1]
    largest_particles = df['Particles'].iloc[-1]

    particle_ratio = largest_particles / base_particles
    time_ratio = largest_time / base_time
    efficiency = (particle_ratio / time_ratio) * 100

    print(f"\nScaling from smallest to largest particle count:")
    print(f"  Particles increased: {particle_ratio:.2f}x ({base_particles:,} → {largest_particles:,})")
    print(f"  Time increased: {time_ratio:.2f}x ({base_time:.2f}s → {largest_time:.2f}s)")
    print(f"  Parallel efficiency: {efficiency:.1f}%")
    print("="*60 + "\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_gpu_scaling.py <results_directory>")
        print("\nExample:")
        print("  python plot_gpu_scaling.py benchmark_gpu_scaling_20251117_120000/")
        sys.exit(1)

    results_dir = sys.argv[1]
    csv_path = os.path.join(results_dir, 'results.csv')

    if not os.path.exists(csv_path):
        print(f"Error: Could not find results.csv in {results_dir}")
        print(f"Expected path: {csv_path}")
        sys.exit(1)

    print(f"Loading results from: {csv_path}")
    df = load_results(csv_path)

    # Calculate derived metrics
    df['Total_Grid_Points'] = df['Grid_NX'] * df['Grid_NY'] * df['Grid_NZ']
    df['Work_Units'] = df['Total_Grid_Points'] * df['Particles']

    print(f"Found {len(df)} benchmark results")
    print("\nData preview:")
    print(df[['Label', 'Particles', 'Time_s', 'Throughput']].to_string(index=False))

    print_summary(df)

    print("\nGenerating plots...")
    plot_gpu_scaling(df, results_dir)

    print("\nDone!")

if __name__ == '__main__':
    main()
