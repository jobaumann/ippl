#!/bin/bash

# Scaling benchmark script for task-parallel vs master branch
# Tests multiple OpenMP thread counts

set -e

# Build directories (can be overridden via environment variables)
TASK_PARALLEL_BUILD_DIR="${TASK_PARALLEL_BUILD_DIR:-build-openmp}"
MASTER_BUILD_DIR="${MASTER_BUILD_DIR:-build-openmp-master}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test parameters
NX=64
NY=64
NZ=64
PARTICLES=100000
TIMESTEPS=10000
SOLVER="FFT"
LBFREQ=10
OVERALLOC=1.0

# Thread counts to test
THREAD_COUNTS=(1 2 4 8 16 32)

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_scaling_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

echo -e "${BLUE}=== IPPL Scaling Benchmark ===${NC}"
echo "Build directories:"
echo "  Task-parallel: ${TASK_PARALLEL_BUILD_DIR}"
echo "  Master: ${MASTER_BUILD_DIR}"
echo ""
echo "Parameters:"
echo "  Grid: ${NX}x${NY}x${NZ}"
echo "  Particles: ${PARTICLES}"
echo "  Timesteps: ${TIMESTEPS}"
echo "  Thread counts: ${THREAD_COUNTS[@]}"
echo ""
echo "Results directory: ${RESULTS_DIR}"
echo ""

# Function to extract timing from output
extract_time() {
    local file=$1
    grep "Elapsed time:" $file | awk '{print $3}'
}

# Function to extract task parallel loop time
extract_task_parallel_time() {
    local file=$1
    grep "taskParallelLoop" $file | awk '{print $3}' || echo "N/A"
}

# Arrays to store results
declare -A TASK_PARALLEL_TIMES
declare -A MASTER_TIMES

# Build both branches first
echo -e "${GREEN}Building task_parallel branch...${NC}"
cd ${SCRIPT_DIR}/${TASK_PARALLEL_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

echo -e "${GREEN}Building master branch...${NC}"
CURRENT_BRANCH=$(git branch --show-current)
git stash
git checkout master
cd ${SCRIPT_DIR}/${MASTER_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}
git checkout ${CURRENT_BRANCH}
git stash pop || true

# Run benchmarks for each thread count
for THREADS in "${THREAD_COUNTS[@]}"; do
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Testing with ${THREADS} OpenMP threads${NC}"
    echo -e "${YELLOW}========================================${NC}"

    export OMP_NUM_THREADS=${THREADS}

    # Run task-parallel branch
    echo -e "${GREEN}Running task-parallel branch with ${THREADS} threads...${NC}"
    cd ${SCRIPT_DIR}/${TASK_PARALLEL_BUILD_DIR}/alpine/ExamplesWithoutPicManager
    ./IndependentParticlesTest ${NX} ${NY} ${NZ} ${PARTICLES} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/task_parallel_${THREADS}threads.txt 2>&1
    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/task_parallel_${THREADS}threads_timing.dat
    fi
    cd ${SCRIPT_DIR}

    TASK_PARALLEL_TIMES[${THREADS}]=$(extract_time ${RESULTS_DIR}/task_parallel_${THREADS}threads.txt)
    echo -e "${GREEN}  Task-parallel (${THREADS} threads): ${TASK_PARALLEL_TIMES[${THREADS}]} seconds${NC}"

    # Run master branch
    echo -e "${GREEN}Running master branch with ${THREADS} threads...${NC}"
    cd ${SCRIPT_DIR}/${MASTER_BUILD_DIR}/alpine/ExamplesWithoutPicManager
    ./IndependentParticlesTest ${NX} ${NY} ${NZ} ${PARTICLES} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/master_${THREADS}threads.txt 2>&1
    if [ -f timing.dat ]; then
        cp timing.dat ${RESULTS_DIR}/master_${THREADS}threads_timing.dat
    fi
    cd ${SCRIPT_DIR}

    MASTER_TIMES[${THREADS}]=$(extract_time ${RESULTS_DIR}/master_${THREADS}threads.txt)
    echo -e "${GREEN}  Master (${THREADS} threads): ${MASTER_TIMES[${THREADS}]} seconds${NC}"

    # Calculate speedup for this thread count
    if [ "${MASTER_TIMES[${THREADS}]}" != "" ] && [ "${TASK_PARALLEL_TIMES[${THREADS}]}" != "" ]; then
        SPEEDUP=$(echo "scale=2; ${MASTER_TIMES[${THREADS}]} / ${TASK_PARALLEL_TIMES[${THREADS}]}" | bc)
        echo -e "${BLUE}  Speedup (task-parallel vs master): ${SPEEDUP}x${NC}"
    fi
    echo ""
done

# Generate summary report
echo -e "${BLUE}=== SCALING BENCHMARK RESULTS ===${NC}"
echo ""
printf "%-10s %-20s %-20s %-15s\n" "Threads" "Task-Parallel (s)" "Master (s)" "Speedup"
printf "%-10s %-20s %-20s %-15s\n" "-------" "------------------" "----------" "-------"

for THREADS in "${THREAD_COUNTS[@]}"; do
    TP_TIME=${TASK_PARALLEL_TIMES[${THREADS}]}
    M_TIME=${MASTER_TIMES[${THREADS}]}
    if [ "$M_TIME" != "" ] && [ "$TP_TIME" != "" ]; then
        SPEEDUP=$(echo "scale=2; ${M_TIME} / ${TP_TIME}" | bc)
        printf "%-10s %-20s %-20s %-15s\n" "${THREADS}" "${TP_TIME}" "${M_TIME}" "${SPEEDUP}x"
    fi
done

echo ""
echo -e "${BLUE}Parallel Efficiency Analysis:${NC}"
echo ""

# Calculate parallel efficiency for task-parallel branch
BASE_TP_TIME=${TASK_PARALLEL_TIMES[1]}
printf "Task-Parallel Scaling:\n"
printf "%-10s %-15s %-15s\n" "Threads" "Time (s)" "Efficiency"
printf "%-10s %-15s %-15s\n" "-------" "--------" "----------"
for THREADS in "${THREAD_COUNTS[@]}"; do
    TP_TIME=${TASK_PARALLEL_TIMES[${THREADS}]}
    if [ "$TP_TIME" != "" ] && [ "$BASE_TP_TIME" != "" ]; then
        SPEEDUP=$(echo "scale=2; ${BASE_TP_TIME} / ${TP_TIME}" | bc)
        EFFICIENCY=$(echo "scale=2; 100 * ${SPEEDUP} / ${THREADS}" | bc)
        printf "%-10s %-15s %-15s%%\n" "${THREADS}" "${TP_TIME}" "${EFFICIENCY}"
    fi
done

echo ""

# Calculate parallel efficiency for master branch
BASE_M_TIME=${MASTER_TIMES[1]}
printf "Master Scaling:\n"
printf "%-10s %-15s %-15s\n" "Threads" "Time (s)" "Efficiency"
printf "%-10s %-15s %-15s\n" "-------" "--------" "----------"
for THREADS in "${THREAD_COUNTS[@]}"; do
    M_TIME=${MASTER_TIMES[${THREADS}]}
    if [ "$M_TIME" != "" ] && [ "$BASE_M_TIME" != "" ]; then
        SPEEDUP=$(echo "scale=2; ${BASE_M_TIME} / ${M_TIME}" | bc)
        EFFICIENCY=$(echo "scale=2; 100 * ${SPEEDUP} / ${THREADS}" | bc)
        printf "%-10s %-15s %-15s%%\n" "${THREADS}" "${M_TIME}" "${EFFICIENCY}"
    fi
done

echo ""
echo "Full results saved in: ${RESULTS_DIR}"

# Save results to CSV for easy plotting
cat > ${RESULTS_DIR}/results.csv <<EOF
Threads,TaskParallel_Time,Master_Time,Speedup
EOF

for THREADS in "${THREAD_COUNTS[@]}"; do
    TP_TIME=${TASK_PARALLEL_TIMES[${THREADS}]}
    M_TIME=${MASTER_TIMES[${THREADS}]}
    if [ "$M_TIME" != "" ] && [ "$TP_TIME" != "" ]; then
        SPEEDUP=$(echo "scale=4; ${M_TIME} / ${TP_TIME}" | bc)
        echo "${THREADS},${TP_TIME},${M_TIME},${SPEEDUP}" >> ${RESULTS_DIR}/results.csv
    fi
done

echo "CSV results saved to: ${RESULTS_DIR}/results.csv"
