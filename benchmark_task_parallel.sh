#!/bin/bash

# Benchmark script for task-parallel vs master branch
# Target: ~5 minutes runtime on 1 CPU core with OpenMP

set -e

# Build directories (can be overridden via environment variables)
TASK_PARALLEL_BUILD_DIR="${TASK_PARALLEL_BUILD_DIR:-build-openmp}"
MASTER_BUILD_DIR="${MASTER_BUILD_DIR:-build-openmp-master}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test parameters - tuned for ~1 minute runtime total
# Reducing from 3000 to 500 timesteps for faster benchmark
NX=64
NY=64
NZ=64
PARTICLES=100000
TIMESTEPS=10000
SOLVER="FFT"
LBFREQ=10
OVERALLOC=1.0

# Force 1 OpenMP thread
export OMP_NUM_THREADS=16

echo -e "${BLUE}=== IPPL Task-Parallel Benchmark ===${NC}"
echo "Build directories:"
echo "  Task-parallel: ${TASK_PARALLEL_BUILD_DIR}"
echo "  Master: ${MASTER_BUILD_DIR}"
echo ""
echo "Parameters:"
echo "  Grid: ${NX}x${NY}x${NZ}"
echo "  Particles: ${PARTICLES}"
echo "  Timesteps: ${TIMESTEPS}"
echo "  OpenMP Threads: ${OMP_NUM_THREADS}"
echo ""

# Create results directory with absolute path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${RESULTS_DIR}

# Function to extract timing from output
extract_time() {
    local file=$1
    # Extract the "Elapsed time" from the output
    grep "Elapsed time:" $file | awk '{print $3}'
}

# Function to extract task parallel loop time
extract_task_parallel_time() {
    local file=$1
    # Extract taskParallelLoop time from timing.dat
    grep "taskParallelLoop" $file | awk '{print $3}' || echo "N/A"
}

# Build task_parallel branch
echo -e "${GREEN}Building task_parallel branch...${NC}"
cd ${SCRIPT_DIR}
cd ${TASK_PARALLEL_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

# Run task_parallel benchmark
echo -e "${GREEN}Running benchmark on task_parallel branch...${NC}"
cd ${SCRIPT_DIR}/${TASK_PARALLEL_BUILD_DIR}/alpine/ExamplesWithoutPicManager
./IndependentParticlesTest ${NX} ${NY} ${NZ} ${PARTICLES} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/task_parallel_output.txt 2>&1
if [ -f timing.dat ]; then
    cp timing.dat ${RESULTS_DIR}/task_parallel_timing.dat
fi
cd ${SCRIPT_DIR}

TASK_PARALLEL_TIME=$(extract_time ${RESULTS_DIR}/task_parallel_output.txt)
TASK_PARALLEL_LOOP_TIME=$(extract_task_parallel_time ${RESULTS_DIR}/task_parallel_timing.dat 2>/dev/null || echo "N/A")
echo -e "${GREEN}Task-parallel elapsed time: ${TASK_PARALLEL_TIME} seconds${NC}"
echo -e "${GREEN}Task-parallel loop time: ${TASK_PARALLEL_LOOP_TIME} seconds${NC}"

# Switch to master branch and build
echo -e "${GREEN}Building master branch...${NC}"
cd ${SCRIPT_DIR}
CURRENT_BRANCH=$(git branch --show-current)
git stash
git checkout master
cd ${MASTER_BUILD_DIR}
cmake --build . -j$(nproc) --target IndependentParticlesTest
cd ${SCRIPT_DIR}

# Run master benchmark
echo -e "${GREEN}Running benchmark on master branch...${NC}"
cd ${SCRIPT_DIR}/${MASTER_BUILD_DIR}/alpine/ExamplesWithoutPicManager
./IndependentParticlesTest ${NX} ${NY} ${NZ} ${PARTICLES} ${TIMESTEPS} ${SOLVER} ${LBFREQ} --overallocate ${OVERALLOC} --info 10 > ${RESULTS_DIR}/master_output.txt 2>&1
if [ -f timing.dat ]; then
    cp timing.dat ${RESULTS_DIR}/master_timing.dat
fi
cd ${SCRIPT_DIR}

MASTER_TIME=$(extract_time ${RESULTS_DIR}/master_output.txt)
echo -e "${GREEN}Master elapsed time: ${MASTER_TIME} seconds${NC}"

# Switch back to original branch
cd ${SCRIPT_DIR}
git checkout ${CURRENT_BRANCH}
git stash pop || true

# Calculate speedup
echo ""
echo -e "${BLUE}=== BENCHMARK RESULTS ===${NC}"
echo "Task-parallel branch: ${TASK_PARALLEL_TIME} seconds"
echo "Master branch:        ${MASTER_TIME} seconds"

if [ "$MASTER_TIME" != "" ] && [ "$TASK_PARALLEL_TIME" != "" ]; then
    SPEEDUP=$(echo "scale=2; ${MASTER_TIME} / ${TASK_PARALLEL_TIME}" | bc)
    PERCENT=$(echo "scale=2; (${MASTER_TIME} - ${TASK_PARALLEL_TIME}) / ${MASTER_TIME} * 100" | bc)
    echo ""
    echo -e "${GREEN}Speedup: ${SPEEDUP}x${NC}"
    if (( $(echo "${SPEEDUP} > 1" | bc -l) )); then
        echo -e "${GREEN}Task-parallel is ${PERCENT}% faster${NC}"
    else
        echo -e "${RED}Task-parallel is ${PERCENT}% slower${NC}"
    fi
fi

echo ""
echo "Full results saved in: ${RESULTS_DIR}"
echo "  - task_parallel_output.txt"
echo "  - task_parallel_timing.dat"
echo "  - master_output.txt"
echo "  - master_timing.dat"

# Save summary
cat > ${RESULTS_DIR}/summary.txt <<EOF
IPPL Task-Parallel Benchmark Results
=====================================
Date: $(date)
Parameters:
  Grid: ${NX}x${NY}x${NZ}
  Particles: ${PARTICLES}
  Timesteps: ${TIMESTEPS}
  OpenMP Threads: ${OMP_NUM_THREADS}

Results:
  Task-parallel: ${TASK_PARALLEL_TIME} seconds
  Master:        ${MASTER_TIME} seconds
  Speedup:       ${SPEEDUP}x
  Improvement:   ${PERCENT}%
EOF

echo ""
echo "Summary saved to: ${RESULTS_DIR}/summary.txt"
