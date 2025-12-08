# Reference Implementations

This directory contains reference implementations from [rammann/ippl-fork](https://github.com/rammann/ippl-fork) to guide the time-blocking birth/death implementation.

## Files

### IndependentParticlesTest_loadbalance.cpp
**Source**: [rammann/ippl-fork/independent-particles](https://github.com/rammann/ippl-fork/blob/independent-particles/alpine/ExamplesWithoutPicManager/IndependentParticlesTest.cpp)

Contains tree-based load balancing implementation:
- `Node<size_type>` class: Tree node structure for hierarchical communication
- `Tree` class: Binary-ternary hybrid tree construction
- `ParticleTreeLayout`: Load balancing with 3 phases:
  1. Work reduction (bottom-up)
  2. Quota computation (top-down)
  3. Work exchange (particle migration)

Key features:
- Hierarchical MPI communication tree
- Hash-based particle selection for migration
- Asynchronous sends with MPI_Isend
- Validity tracking for particle deletion after sending

### ParticleContainerTest.cpp
**Source**: [rammann/ippl-fork/particle-containers](https://github.com/rammann/ippl-fork/tree/particle-containers/alpine/ExamplesWithoutPicManager/ParticleContainerTest.cpp)

Contains particle birth/death functions:
- `death()`: Probabilistic particle deletion using boolean mask
- `birth()`: Probabilistic particle creation with parallel_reduce counting
- Species tracking using attribute-based or index-range approaches

Key features:
- Kokkos parallel patterns for GPU acceleration
- Thread-safe random number generation via pool
- Boolean view for marking particles to destroy
- Separate create/destroy calls instead of dormant pool

## Usage

These files provide reference implementations for:
1. Tree-based load balancing in Stage 3 of time-blocking
2. Proper birth/death tracking using create/destroy instead of Qview manipulation
