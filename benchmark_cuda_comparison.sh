#!/bin/bash

# CUDA Branch Comparison Benchmark
# Compares task-parallel vs master branch on CUDA with varying particle counts

set -e

# Build directories (can be overridden via environment variables)
TASK_PARALLEL_BUILD_DIR="${TASK_PARALLEL_BUILD_DIR:-build-cuda}"
MASTER_BUILD_DIR="${MASTER_BUILD_DIR:-build-cuda-master}"

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

# Fixed grid size (grid size has no impact on performance)
NX=128
NY=128
NZ=128

# Particle counts to test
# Format: "PARTICLES LABEL"
declare -a PARTICLE_COUNTS=(
    "2500 2.5K"
    "5000 5K"
    "10000 10K"
    "25000 25K"
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
RESULTS_DIR="${SCRIPT_DIR}/benchmark_cuda_comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo -e "${CYAN}=== IPPL CUDA Branch Comparison Benchmark ===${NC}"
echo "Build directories:"
echo "  Task-parallel: ${TASK_PARALLEL_BUILD_DIR}"
echo "  Master:        ${MASTER_BUILD_DIR}"
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

# Arrays to store results
declare -A TASK_PARALLEL_TIMES
declare -A MASTER_TIMES
declare -A PARTICLE_VALUES

# Build both versions first
echo -e "${GREEN}Building master branch...${NC}"
cd ${SCRIPT_DIR}/${MASTER_BUILD_DIR}
git checkout master
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

echo -e "${GREEN}Building task-parallel branch...${NC}"
cd ${SCRIPT_DIR}/${TASK_PARALLEL_BUILD_DIR}
git checkout task_parallel
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

echo ""

# Run benchmarks for each particle count
for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"

    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing particle count: ${label}${NC}"
    echo -e "${YELLOW}Grid: ${NX}x${NY}x${NZ}, Particles: ${particles}${NC}"
    echo -e "${YELLOW}========================================${NC}"

    # Run task-parallel benchmark
    echo -e "${BLUE}Running task-parallel branch for ${label}...${NC}"
    cd ${SCRIPT_DIR}/${TASK_PARALLEL_BUILD_DIR}/alpine/ExamplesWithoutPicManager

    ./IndependentParticlesTest ${NX} ${NY} ${NZ} ${particles} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/task_parallel_${label}.txt 2>&1

    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/task_parallel_${label}_timing.dat
    fi
    cd ${SCRIPT_DIR}

    TASK_PARALLEL_TIMES[${label}]=$(extract_time ${RESULTS_DIR}/task_parallel_${label}.txt)
    echo -e "${BLUE}  Task-parallel: ${TASK_PARALLEL_TIMES[${label}]} seconds${NC}"

    # Run master benchmark
    echo -e "${GREEN}Running master branch for ${label}...${NC}"
    cd ${SCRIPT_DIR}/${MASTER_BUILD_DIR}/alpine/ExamplesWithoutPicManager

    ./IndependentParticlesTest ${NX} ${NY} ${NZ} ${particles} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/master_${label}.txt 2>&1

    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/master_${label}_timing.dat
    fi
    cd ${SCRIPT_DIR}

    MASTER_TIMES[${label}]=$(extract_time ${RESULTS_DIR}/master_${label}.txt)
    echo -e "${GREEN}  Master:        ${MASTER_TIMES[${label}]} seconds${NC}"

    # Store metadata
    PARTICLE_VALUES[${label}]=${particles}

    # Calculate speedup
    task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
    master_time=${MASTER_TIMES[${label}]}

    if [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "" ] && [ "$task_parallel_time" != "" ]; then
        speedup=$(echo "scale=2; ${master_time} / ${task_parallel_time}" | bc)
        if (( $(echo "${speedup} >= 1" | bc -l) )); then
            echo -e "${CYAN}  Speedup: ${speedup}x (task-parallel is faster)${NC}"
        else
            slowdown=$(echo "scale=2; ${task_parallel_time} / ${master_time}" | bc)
            echo -e "${RED}  Speedup: ${speedup}x (master is ${slowdown}x faster)${NC}"
        fi
    fi
    echo ""
done

# Generate summary report
echo -e "${CYAN}=== CUDA BRANCH COMPARISON RESULTS ===${NC}"
echo ""
printf "%-12s %-15s %-18s %-18s %-12s\n" "Label" "Particles" "Task-parallel (s)" "Master (s)" "Speedup"
printf "%-12s %-15s %-18s %-18s %-12s\n" "-----" "---------" "------------------" "----------" "-------"

for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
    master_time=${MASTER_TIMES[${label}]}

    if [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "" ] && [ "$task_parallel_time" != "" ]; then
        speedup=$(echo "scale=2; ${master_time} / ${task_parallel_time}" | bc)
        printf "%-12s %-15s %-18s %-18s %-12s\n" "${label}" "${particles}" "${task_parallel_time}" "${master_time}" "${speedup}x"
    else
        printf "%-12s %-15s %-18s %-18s %-12s\n" "${label}" "${particles}" "${task_parallel_time}" "${master_time}" "N/A"
    fi
done

echo ""
echo -e "${CYAN}Performance Analysis:${NC}"
echo ""

# Calculate average speedup
total_speedup=0
count=0
for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
    master_time=${MASTER_TIMES[${label}]}

    if [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "" ] && [ "$task_parallel_time" != "" ]; then
        speedup=$(echo "scale=4; ${master_time} / ${task_parallel_time}" | bc)
        total_speedup=$(echo "scale=4; ${total_speedup} + ${speedup}" | bc)
        count=$((count + 1))
    fi
done

if [ $count -gt 0 ]; then
    avg_speedup=$(echo "scale=2; ${total_speedup} / ${count}" | bc)
    echo "Average Task-Parallel Speedup: ${avg_speedup}x"
    echo ""
fi

# Analyze scaling
echo "Scaling with Particle Count:"
printf "%-12s %-20s %-20s\n" "Label" "Task-parallel (s)" "Master (s)"
printf "%-12s %-20s %-20s\n" "-----" "-----------------" "---------"

base_label="50K"
base_task_parallel=${TASK_PARALLEL_TIMES[${base_label}]}
base_master=${MASTER_TIMES[${base_label}]}

if [ "$base_task_parallel" != "N/A" ] && [ "$base_master" != "N/A" ] && [ "$base_task_parallel" != "" ] && [ "$base_master" != "" ]; then
    for pcount in "${PARTICLE_COUNTS[@]}"; do
        read -r particles label <<< "$pcount"
        task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
        master_time=${MASTER_TIMES[${label}]}

        if [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "" ] && [ "$master_time" != "" ]; then
            task_parallel_ratio=$(echo "scale=2; ${task_parallel_time} / ${base_task_parallel}" | bc)
            master_ratio=$(echo "scale=2; ${master_time} / ${base_master}" | bc)
            printf "%-12s %-20s %-20s\n" "${label}" "${task_parallel_ratio}x" "${master_ratio}x"
        fi
    done
fi

echo ""
echo "Full results saved in: ${RESULTS_DIR}"

# Save results to CSV for easy plotting
cat > ${RESULTS_DIR}/results.csv <<EOF
Label,Particles,TaskParallel_Time_s,Master_Time_s,Speedup,TaskParallel_Throughput,Master_Throughput
EOF

for pcount in "${PARTICLE_COUNTS[@]}"; do
    read -r particles label <<< "$pcount"
    task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
    master_time=${MASTER_TIMES[${label}]}

    if [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "" ] && [ "$master_time" != "" ]; then
        speedup=$(echo "scale=4; ${master_time} / ${task_parallel_time}" | bc)
        total_updates=$(echo "${particles} * ${TIMESTEPS}" | bc)
        task_parallel_throughput=$(echo "scale=2; ${total_updates} / ${task_parallel_time}" | bc)
        master_throughput=$(echo "scale=2; ${total_updates} / ${master_time}" | bc)
        echo "${label},${particles},${task_parallel_time},${master_time},${speedup},${task_parallel_throughput},${master_throughput}" >> ${RESULTS_DIR}/results.csv
    fi
done

echo "CSV results saved to: ${RESULTS_DIR}/results.csv"

# Create a summary file
cat > ${RESULTS_DIR}/summary.txt <<EOF
IPPL CUDA Branch Comparison Benchmark Results
==============================================
Date: $(date)
Task-Parallel Build: ${TASK_PARALLEL_BUILD_DIR}
Master Build:        ${MASTER_BUILD_DIR}

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
    task_parallel_time=${TASK_PARALLEL_TIMES[${label}]}
    master_time=${MASTER_TIMES[${label}]}

    if [ "$task_parallel_time" != "N/A" ] && [ "$master_time" != "N/A" ] && [ "$task_parallel_time" != "" ] && [ "$master_time" != "" ]; then
        speedup=$(echo "scale=2; ${master_time} / ${task_parallel_time}" | bc)
    else
        speedup="N/A"
    fi

    cat >> ${RESULTS_DIR}/summary.txt <<EOF

${label}:
  Particles: ${particles}
  Task-Parallel Time: ${task_parallel_time} seconds
  Master Time:        ${master_time} seconds
  Speedup:            ${speedup}x
EOF
done

echo ""
echo "Summary saved to: ${RESULTS_DIR}/summary.txt"
