#!/bin/bash

# GPU vs CPU Comparison Benchmark
# Compares CUDA and OpenMP performance across different problem sizes

set -e

# Build directories (can be overridden via environment variables)
CUDA_BUILD_DIR="${CUDA_BUILD_DIR:-build-cuda}"
CPU_BUILD_DIR="${CPU_BUILD_DIR:-build-openmp}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Fixed parameters
TIMESTEPS=1000
SOLVER="FFT"
LBFREQ=10
OVERALLOC=1.0

# CPU threads to use
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}

# Problem sizes to test
# Format: "NX NY NZ PARTICLES LABEL"
declare -a PROBLEM_SIZES=(
    "32 32 32 50000 small"
    "64 64 64 100000 medium"
    "96 96 96 250000 large"
    "128 128 128 500000 xlarge"
)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_gpu_vs_cpu_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo -e "${CYAN}=== IPPL GPU vs CPU Comparison Benchmark ===${NC}"
echo "Build directories:"
echo "  CUDA: ${CUDA_BUILD_DIR}"
echo "  CPU:  ${CPU_BUILD_DIR}"
echo ""
echo "Fixed parameters:"
echo "  Timesteps: ${TIMESTEPS}"
echo "  Solver: ${SOLVER}"
echo "  CPU Threads: ${OMP_NUM_THREADS}"
echo ""
echo "Problem sizes to test:"
for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"
    echo "  ${label}: ${nx}x${ny}x${nz} grid, ${particles} particles"
done
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Function to extract timing from output
extract_time() {
    local file=$1
    grep "Elapsed time:" $file | awk '{print $3}' || echo "N/A"
}

# Arrays to store results
declare -A CUDA_TIMES
declare -A CPU_TIMES
declare -A GRID_SIZES
declare -A PARTICLE_COUNTS

# Build both versions first
echo -e "${GREEN}Building CUDA version...${NC}"
cd ${SCRIPT_DIR}/${CUDA_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

echo -e "${GREEN}Building CPU/OpenMP version...${NC}"
cd ${SCRIPT_DIR}/${CPU_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

echo ""

# Run benchmarks for each problem size
for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing problem size: ${label}${NC}"
    echo -e "${YELLOW}Grid: ${nx}x${ny}x${nz}, Particles: ${particles}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Run CUDA benchmark
    echo -e "${BLUE}Running CUDA benchmark for ${label}...${NC}"
    cd ${SCRIPT_DIR}/${CUDA_BUILD_DIR}/alpine/ExamplesWithoutPicManager

    ./IndependentParticlesTest ${nx} ${ny} ${nz} ${particles} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/cuda_${label}.txt 2>&1

    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/cuda_${label}_timing.dat
    fi
    cd ${SCRIPT_DIR}

    CUDA_TIMES[${label}]=$(extract_time ${RESULTS_DIR}/cuda_${label}.txt)
    echo -e "${BLUE}  CUDA: ${CUDA_TIMES[${label}]} seconds${NC}"

    # Run CPU benchmark
    echo -e "${GREEN}Running CPU/OpenMP benchmark for ${label}...${NC}"
    cd ${SCRIPT_DIR}/${CPU_BUILD_DIR}/alpine/ExamplesWithoutPicManager

    ./IndependentParticlesTest ${nx} ${ny} ${nz} ${particles} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/cpu_${label}.txt 2>&1

    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/cpu_${label}_timing.dat
    fi
    cd ${SCRIPT_DIR}

    CPU_TIMES[${label}]=$(extract_time ${RESULTS_DIR}/cpu_${label}.txt)
    echo -e "${GREEN}  CPU:  ${CPU_TIMES[${label}]} seconds${NC}"

    # Store metadata
    GRID_SIZES[${label}]="${nx}x${ny}x${nz}"
    PARTICLE_COUNTS[${label}]=${particles}

    # Calculate speedup
    cuda_time=${CUDA_TIMES[${label}]}
    cpu_time=${CPU_TIMES[${label}]}

    if [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "" ] && [ "$cuda_time" != "" ]; then
        speedup=$(echo "scale=2; ${cpu_time} / ${cuda_time}" | bc)
        if (( $(echo "${speedup} >= 1" | bc -l) )); then
            echo -e "${CYAN}  Speedup: ${speedup}x (GPU is faster)${NC}"
        else
            slowdown=$(echo "scale=2; ${cuda_time} / ${cpu_time}" | bc)
            echo -e "${RED}  Speedup: ${speedup}x (CPU is ${slowdown}x faster)${NC}"
        fi
    fi
    echo ""
done

# Generate summary report
echo -e "${CYAN}=== GPU vs CPU BENCHMARK RESULTS ===${NC}"
echo ""
printf "%-12s %-15s %-15s %-15s %-15s %-12s\n" "Problem" "Grid Size" "Particles" "CUDA (s)" "CPU (s)" "Speedup"
printf "%-12s %-15s %-15s %-15s %-15s %-12s\n" "--------" "---------" "---------" "--------" "-------" "-------"

for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"
    cuda_time=${CUDA_TIMES[${label}]}
    cpu_time=${CPU_TIMES[${label}]}
    grid=${GRID_SIZES[${label}]}

    if [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "" ] && [ "$cuda_time" != "" ]; then
        speedup=$(echo "scale=2; ${cpu_time} / ${cuda_time}" | bc)
        printf "%-12s %-15s %-15s %-15s %-15s %-12s\n" "${label}" "${grid}" "${particles}" "${cuda_time}" "${cpu_time}" "${speedup}x"
    else
        printf "%-12s %-15s %-15s %-15s %-15s %-12s\n" "${label}" "${grid}" "${particles}" "${cuda_time}" "${cpu_time}" "N/A"
    fi
done

echo ""
echo -e "${CYAN}Performance Analysis:${NC}"
echo ""

# Calculate average speedup
total_speedup=0
count=0
for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"
    cuda_time=${CUDA_TIMES[${label}]}
    cpu_time=${CPU_TIMES[${label}]}

    if [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "" ] && [ "$cuda_time" != "" ]; then
        speedup=$(echo "scale=4; ${cpu_time} / ${cuda_time}" | bc)
        total_speedup=$(echo "scale=4; ${total_speedup} + ${speedup}" | bc)
        count=$((count + 1))
    fi
done

if [ $count -gt 0 ]; then
    avg_speedup=$(echo "scale=2; ${total_speedup} / ${count}" | bc)
    echo "Average GPU Speedup: ${avg_speedup}x"
    echo ""
fi

# Analyze scaling
echo "Scaling with Problem Size:"
printf "%-12s %-15s %-15s\n" "Problem" "CUDA scaling" "CPU scaling"
printf "%-12s %-15s %-15s\n" "--------" "-------------" "-----------"

base_label="small"
base_cuda=${CUDA_TIMES[${base_label}]}
base_cpu=${CPU_TIMES[${base_label}]}

if [ "$base_cuda" != "N/A" ] && [ "$base_cpu" != "N/A" ] && [ "$base_cuda" != "" ] && [ "$base_cpu" != "" ]; then
    for size in "${PROBLEM_SIZES[@]}"; do
        read -r nx ny nz particles label <<< "$size"
        cuda_time=${CUDA_TIMES[${label}]}
        cpu_time=${CPU_TIMES[${label}]}

        if [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "" ] && [ "$cpu_time" != "" ]; then
            cuda_ratio=$(echo "scale=2; ${cuda_time} / ${base_cuda}" | bc)
            cpu_ratio=$(echo "scale=2; ${cpu_time} / ${base_cpu}" | bc)
            printf "%-12s %-15s %-15s\n" "${label}" "${cuda_ratio}x" "${cpu_ratio}x"
        fi
    done
fi

echo ""
echo "Full results saved in: ${RESULTS_DIR}"

# Save results to CSV for easy plotting
cat > ${RESULTS_DIR}/results.csv <<EOF
Label,Grid_NX,Grid_NY,Grid_NZ,Particles,CUDA_Time_s,CPU_Time_s,Speedup,CUDA_Throughput,CPU_Throughput
EOF

for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"
    cuda_time=${CUDA_TIMES[${label}]}
    cpu_time=${CPU_TIMES[${label}]}

    if [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "" ] && [ "$cpu_time" != "" ]; then
        speedup=$(echo "scale=4; ${cpu_time} / ${cuda_time}" | bc)
        total_updates=$(echo "${particles} * ${TIMESTEPS}" | bc)
        cuda_throughput=$(echo "scale=2; ${total_updates} / ${cuda_time}" | bc)
        cpu_throughput=$(echo "scale=2; ${total_updates} / ${cpu_time}" | bc)
        echo "${label},${nx},${ny},${nz},${particles},${cuda_time},${cpu_time},${speedup},${cuda_throughput},${cpu_throughput}" >> ${RESULTS_DIR}/results.csv
    fi
done

echo "CSV results saved to: ${RESULTS_DIR}/results.csv"

# Create a summary file
cat > ${RESULTS_DIR}/summary.txt <<EOF
IPPL GPU vs CPU Comparison Benchmark Results
=============================================
Date: $(date)
CUDA Build: ${CUDA_BUILD_DIR}
CPU Build:  ${CPU_BUILD_DIR}
CPU Threads: ${OMP_NUM_THREADS}

Fixed Parameters:
  Timesteps: ${TIMESTEPS}
  Solver: ${SOLVER}
  Load Balance Frequency: ${LBFREQ}
  Overallocation: ${OVERALLOC}

Results:
EOF

for size in "${PROBLEM_SIZES[@]}"; do
    read -r nx ny nz particles label <<< "$size"
    cuda_time=${CUDA_TIMES[${label}]}
    cpu_time=${CPU_TIMES[${label}]}

    if [ "$cuda_time" != "N/A" ] && [ "$cpu_time" != "N/A" ] && [ "$cuda_time" != "" ] && [ "$cpu_time" != "" ]; then
        speedup=$(echo "scale=2; ${cpu_time} / ${cuda_time}" | bc)
    else
        speedup="N/A"
    fi

    cat >> ${RESULTS_DIR}/summary.txt <<EOF

${label}:
  Grid: ${nx}x${ny}x${nz}
  Particles: ${particles}
  CUDA Time: ${cuda_time} seconds
  CPU Time:  ${cpu_time} seconds
  Speedup:   ${speedup}x
EOF
done

echo ""
echo "Summary saved to: ${RESULTS_DIR}/summary.txt"
