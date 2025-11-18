#!/bin/bash

# GPU Problem Size Scaling Benchmark
# Tests CUDA performance with varying problem sizes

set -e

# Build directory (can be overridden via environment variable)
CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-build-cuda-master}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Fixed parameters
TIMESTEPS=1000
SOLVER="FFT"
LBFREQ=10
OVERALLOC=1.0

# Fixed grid size (grid size has no impact on performance)
NX=128
NY=128
NZ=128

# Particle counts to test
# Format: "PARTICLES LABEL"
declare -a PARTICLE_COUNTS=(
    "50000 50K"
    "100000 100K"
    "250000 250K"
    "500000 500K"
    "1000000 1M"
    "2000000 2M"
    "5000000 5M"
)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_gpu_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo -e "${BLUE}=== IPPL GPU Particle Scaling Benchmark ===${NC}"
echo "Build directory: ${CUDA_BUILD_DIR}"
echo ""
echo "Fixed parameters:"
echo "  Grid size: ${NX}x${NY}x${NZ}"
echo "  Timesteps: ${TIMESTEPS}"
echo "  Solver: ${SOLVER}"
echo ""
echo "Particle counts to test:"
for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    echo "  ${label}: ${particles} particles"
done
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Function to extract timing from output
extract_time() {
    local file=$1
    grep "Elapsed time:" $file | awk '{print $3}' || echo "N/A"
}

# Function to extract memory usage from output
extract_memory() {
    local file=$1
    grep -i "memory" $file | head -1 || echo "N/A"
}

# Arrays to store results
declare -A TIMES
declare -A PARTICLE_VALUES

# Build CUDA version first
echo -e "${GREEN}Building CUDA version...${NC}"
cd ${SCRIPT_DIR}/${CUDA_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

# Run benchmarks for each particle count
for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing particle count: ${label}${NC}"
    echo -e "${YELLOW}Grid: ${NX}x${NY}x${NZ}, Particles: ${particles}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Run CUDA benchmark
    echo -e "${GREEN}Running CUDA benchmark for ${label}...${NC}"
    cd ${SCRIPT_DIR}/${CUDA_BUILD_DIR}/alpine/ExamplesWithoutPicManager

    ./IndependentParticlesTest ${NX} ${NY} ${NZ} ${particles} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/cuda_${label}.txt 2>&1

    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/cuda_${label}_timing.dat
    fi
    cd ${SCRIPT_DIR}

    TIMES[${label}]=$(extract_time ${RESULTS_DIR}/cuda_${label}.txt)
    PARTICLE_VALUES[${label}]=${particles}

    echo -e "${GREEN}  ${label}: ${TIMES[${label}]} seconds${NC}"
    echo ""
done

# Generate summary report
echo -e "${BLUE}=== GPU SCALING BENCHMARK RESULTS ===${NC}"
echo ""
printf "%-12s %-15s %-15s %-20s\n" "Label" "Particles" "Time (s)" "Throughput (M/s)"
printf "%-12s %-15s %-15s %-20s\n" "-----" "---------" "--------" "-----------------"

for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    time=${TIMES[${label}]}

    if [ "$time" != "N/A" ] && [ "$time" != "" ]; then
        # Calculate particles per second (total particle updates = particles * timesteps)
        total_updates=$(echo "${particles} * ${TIMESTEPS}" | bc)
        throughput=$(echo "scale=2; ${total_updates} / ${time} / 1000000" | bc)
        printf "%-12s %-15s %-15s %-20s\n" "${label}" "${particles}" "${time}" "${throughput}"
    else
        printf "%-12s %-15s %-15s %-20s\n" "${label}" "${particles}" "FAILED" "N/A"
    fi
done

echo ""
echo -e "${BLUE}Performance Scaling Analysis:${NC}"
echo ""

# Calculate speedup relative to smallest problem
base_label="50K"
base_time=${TIMES[${base_label}]}
base_particles=${PARTICLE_VALUES[${base_label}]}

if [ "$base_time" != "N/A" ] && [ "$base_time" != "" ]; then
    printf "%-12s %-15s %-20s %-20s\n" "Label" "Time (s)" "Time vs 50K" "Particles vs 50K"
    printf "%-12s %-15s %-20s %-20s\n" "-----" "--------" "-----------" "----------------"

    for pcount in "${PARTICLE_COUNTS[@]}"; do
        read -r particles label <<< "$pcount"
        time=${TIMES[${label}]}

        if [ "$time" != "N/A" ] && [ "$time" != "" ]; then
            time_ratio=$(echo "scale=2; ${time} / ${base_time}" | bc)
            particle_ratio=$(echo "scale=2; ${particles} / ${base_particles}" | bc)
            printf "%-12s %-15s %-20s %-20s\n" "${label}" "${time}" "${time_ratio}x" "${particle_ratio}x"
        fi
    done
fi

echo ""
echo "Full results saved in: ${RESULTS_DIR}"

# Save results to CSV for easy plotting
cat > ${RESULTS_DIR}/results.csv <<EOF
Label,Grid_NX,Grid_NY,Grid_NZ,Particles,Time_s,Total_Updates,Throughput
EOF

for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    time=${TIMES[${label}]}

    if [ "$time" != "N/A" ] && [ "$time" != "" ]; then
        total_updates=$(echo "${particles} * ${TIMESTEPS}" | bc)
        throughput=$(echo "scale=2; ${total_updates} / ${time}" | bc)
        echo "${label},${NX},${NY},${NZ},${particles},${time},${total_updates},${throughput}" >> ${RESULTS_DIR}/results.csv
    fi
done

echo "CSV results saved to: ${RESULTS_DIR}/results.csv"

# Create a summary file
cat > ${RESULTS_DIR}/summary.txt <<EOF
IPPL GPU Particle Scaling Benchmark Results
============================================
Date: $(date)
Build Directory: ${CUDA_BUILD_DIR}

Fixed Parameters:
  Grid Size: ${NX}x${NY}x${NZ}
  Timesteps: ${TIMESTEPS}
  Solver: ${SOLVER}
  Load Balance Frequency: ${LBFREQ}
  Overallocation: ${OVERALLOC}

Results:
EOF

for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    time=${TIMES[${label}]}

    cat >> ${RESULTS_DIR}/summary.txt <<EOF

${label}:
  Particles: ${particles}
  Time: ${time} seconds
EOF
done

echo ""
echo "Summary saved to: ${RESULTS_DIR}/summary.txt"
