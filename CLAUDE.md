# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Independent Parallel Particle Layer (IPPL) is a performance-portable C++ library for Particle-Mesh methods. It uses Kokkos for performance portability, HeFFTe for FFT operations, and MPI for distributed parallelism. IPPL supports 1-6 dimensional simulations with mixed precision on CPUs and GPUs.

License: GNU GPL version 3 (as of version 3.2.0, due to modified GNU variant header for CUDA 12.2 compatibility).

## Build System

### CMake Configuration

IPPL uses CMake (minimum 3.24) with preset configurations in `CMakeUserPresets.json`.

**Key CMake Options:**
- `IPPL_PLATFORMS`: Platform backends - `SERIAL`, `OPENMP`, `CUDA`, `HIP`, or combinations like `"OPENMP;CUDA"`
- `IPPL_ENABLE_FFT`: Enable FFT support via HeFFTe (default: OFF)
- `IPPL_ENABLE_SOLVERS`: Enable field solvers (default: OFF)
- `IPPL_ENABLE_ALPINE`: Enable Alpine PIC simulation examples (default: OFF)
- `IPPL_ENABLE_TESTS`: Build integration tests (default: OFF)
- `IPPL_ENABLE_UNIT_TESTS`: Build unit tests with GoogleTest (default: OFF)
- `IPPL_USE_ALTERNATIVE_VARIANT`: Use modified variant for CUDA 12.2 + GCC 12.3.0 (default: OFF)
- `CMAKE_BUILD_TYPE`: `Release`, `RelWithDebInfo`, or `Debug` (enables sanitizers)
- `Kokkos_VERSION`: Default `4.5.00` (use `git.X.Y.Z` to force fetch specific version)
- `Heffte_VERSION`: Default `2.4.0` (use `git.vX.Y.Z` to force fetch specific version)
- `Kokkos_ARCH_*`: Target GPU architecture (e.g., `AMPERE80`, `HOPPER90`, `AMD_GFX90A`)

**Useful Presets:**
```bash
# Configure with preset
cmake --preset=release-testing ..
cmake --preset=debug-testing ..
cmake --preset=openmp ..
```

**Typical Build Workflow:**
```bash
# Serial debug build with tests
cmake -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_STANDARD=20 \
      -DIPPL_ENABLE_TESTS=ON \
      -DIPPL_ENABLE_UNIT_TESTS=ON \
      ..

# OpenMP release build with FFT and solvers
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_STANDARD=20 \
      -DIPPL_PLATFORMS=OPENMP \
      -DIPPL_ENABLE_FFT=ON \
      -DIPPL_ENABLE_SOLVERS=ON \
      -DIPPL_ENABLE_ALPINE=ON \
      -DHeffte_ENABLE_FFTW=ON \
      ..

# CUDA build (requires GPU architecture)
cmake -DCMAKE_BUILD_TYPE=Release \
      -DKokkos_ARCH_AMPERE80=ON \
      -DCMAKE_CXX_STANDARD=20 \
      -DIPPL_PLATFORMS=CUDA \
      -DIPPL_ENABLE_FFT=ON \
      -DIPPL_ENABLE_SOLVERS=ON \
      -DIPPL_USE_ALTERNATIVE_VARIANT=ON \
      ..

# Then build
cmake --build . -j$(nproc)
```

### Testing

**Integration Tests:**
Integration tests are full simulation examples in `test/` subdirectories. Run with:
```bash
# Configure with tests enabled
cmake -DIPPL_ENABLE_TESTS=ON ..
make
ctest
# Or run individual test binaries from test/ subdirectories
```

**Unit Tests:**
Unit tests use GoogleTest and are in `unit_tests/`. They test all combinations of:
- Precision: `float`, `double`
- Dimensionality: 1D through 6D
- Execution spaces: Based on enabled Kokkos backends (Serial, OpenMP, CUDA, HIP)

Run unit tests:
```bash
cmake -DIPPL_ENABLE_UNIT_TESTS=ON ..
make
# Run all unit tests
for file in $(find ./unit_tests -type f -executable); do $file; done
# Or use ctest
ctest
```

## Code Architecture

### Core Modules (src/)

**Particle System:**
- `Particle/`: Particle data structures
  - `ParticleBase`: Base class for particle containers
  - `ParticleAttrib`: Particle attributes (positions, velocities, etc.)
  - `ParticleLayout`: Spatial layouts (e.g., `ParticleSpatialLayout`)
  - `ParticleBC`: Boundary conditions for particles

**Field System:**
- `Field/`: Field data structures on grids
  - `BareField`: Raw field data without boundary conditions
  - `Field`: Field with boundary conditions
  - `BConds`, `BcTypes`: Boundary condition types and application
  - `HaloCells`: Halo cell exchange for distributed fields
- `FieldLayout/`: Domain decomposition for fields
  - `FieldLayout`: Describes field distribution across MPI ranks
  - `SubFieldLayout`: Layout for field subregions

**Spatial Structures:**
- `Index/`: Multi-dimensional indexing (`Index`, `NDIndex`)
- `Region/`: Spatial regions (`NDRegion`, `PRegion`, `RegionLayout`)
- `Meshes/`: Mesh representations (`UniformCartesian`, `Mesh`, `Centering`)

**Solvers:**
- `PoissonSolvers/`: Electrostatic field solvers
  - `FFTPeriodicPoissonSolver`: Periodic boundaries via FFT
  - `FFTOpenPoissonSolver`: Open boundaries via FFT
  - `PoissonCG`: Conjugate gradient solver
  - `FEMPoissonSolver`: Finite element method solver
- `MaxwellSolvers/`: Electromagnetic field solvers
  - `StandardFDTDSolver`, `NonStandardFDTDSolver`: FDTD methods
  - `FEMMaxwellDiffusionSolver`: FEM for Maxwell equations
- `LinearSolvers/`: Generic linear solvers (`PCG`, `SolverAlgorithm`, `Preconditioner`)

**FEM Support:**
- `FEM/`: Finite element method infrastructure
  - `FiniteElementSpace`: Abstract FE space
  - `LagrangeSpace`, `NedelecSpace`: Specific element types
  - `Elements/`: Element definitions (Edge, Quadrilateral, Hexahedral)
  - `Quadrature/`: Numerical integration (Gauss-Jacobi, Midpoint)

**Parallelism:**
- `Communicate/`: MPI communication wrappers
  - `Communicator`: MPI communicator abstraction
  - `Environment`: MPI environment initialization
  - `Archive`: Serialization for communication
- `Decomposition/`: Domain decomposition strategies (`OrthogonalRecursiveBisection`)
- `Partition/`: Particle partitioning across ranks

**Utilities:**
- `FFT/`: FFT interface using HeFFTe
- `Random/`: Random number generation (normal, uniform distributions)
- `Interpolation/`: Particle-mesh interpolation (e.g., CIC)
- `Interaction/`: Particle-particle interactions
- `Expression/`: Expression templates for field operations
- `Types/`: Core type definitions (`Vector`, `IpplTypes`, `ViewTypes`)
- `Utility/`: General utilities (`Timer`, `Inform`, `IpplException`)

### Manager Framework

`Manager/` provides high-level simulation orchestration:
- `BaseManager`: Base simulation manager with time-stepping
- `PicManager`: Template for Particle-in-Cell simulations
  - Defines `par2grid()` and `grid2par()` virtual methods
  - Integrates particle container, field container, field solver, and load balancer
- `FieldSolverBase`: Interface for field solvers

### Alpine Module

`alpine/` contains complete PIC simulation examples using `PicManager`:
- `LandauDamping`: Plasma Landau damping simulation
- `BumponTailInstability`: Bump-on-tail instability
- `PenningTrap`: Penning trap simulation
- `ExamplesWithoutPicManager/`: Standalone examples without manager framework

Each Alpine example defines a custom manager inheriting from `PicManager` with specific physics implementations.

## Code Style and Conventions

**Naming:**
- Variables: camelCase (e.g., `fieldLayout`)
- Compile-time constants: CAPITAL_CASE (e.g., `MAX_DIM`)
- Member variables: suffix with `_m` (e.g., `particles_m`)

**Math Functions:**
- Device/host code: Use `Kokkos::sqrt`, `Kokkos::sin`, etc. (mark with `KOKKOS_INLINE_FUNCTION`)
- Host-only code: Use `std::sqrt`, `std::sin`, etc.
- Constants: Use `Kokkos::numbers::pi`, etc.

**Code Formatting:**
- IPPL uses `clang-format` with settings in `.clang-format`
- Line limit: 100 characters
- Indentation: 4 spaces
- Style: Based on Google style with modifications
- Pre-commit hook available in `hooks/` - run `hooks/setup.sh` to install

**Pre-commit Hook Setup:**
```bash
# From repository root
./hooks/setup.sh
# Now clang-format runs automatically before each commit
```

## Important Patterns

### Template Parameters
IPPL heavily uses templates for:
- **Precision**: `T` is typically `float` or `double`
- **Dimensionality**: `Dim` or `unsigned Dim` for 1-6 dimensions
- **Execution Space**: Kokkos execution spaces (e.g., `Kokkos::OpenMP`, `Kokkos::Cuda`)

Example:
```cpp
template <typename T, unsigned Dim>
class Field { /* ... */ };

// Used as:
Field<double, 3> electricField;  // 3D double-precision field
```

### Kokkos Views
Fields and particle attributes are stored in `Kokkos::View`:
- Host-accessible: `Kokkos::View<T*, Kokkos::HostSpace>`
- Device-accessible: `Kokkos::View<T*, Kokkos::DefaultExecutionSpace>`
- Mirror views used for host-device transfers

### Unit Test Infrastructure
Unit tests auto-generate all combinations of template parameters (precision × dimension × execution space) using `TestParams` in `unit_tests/TestUtils.h`. Tests use GoogleTest's `TYPED_TEST_CASE` to instantiate parameterized tests.

## Development Workflow

1. **Make changes**: Edit source files
2. **Format code**: Either use pre-commit hook or manually run `clang-format`
3. **Build**: `cmake --build . -j$(nproc)`
4. **Test**: Run `ctest` or individual test executables
5. **Commit**: Follow formatting guidelines in `WORKFLOW.md`

For unit tests, see detailed info in `UNIT_TESTS.md`.

## Dependencies

- **CMake** 3.24+
- **C++ Compiler** with C++20 support (GPU-capable for GPU builds)
- **MPI** (GPU-aware for GPU builds)
- **Kokkos**: Performance portability layer (auto-fetched by CMake)
- **HeFFTe**: FFT library (auto-fetched when FFT enabled)
- **Optional**: FFTW, CuFFT for FFT backends
- **Optional**: GoogleTest (auto-fetched for unit tests)

CMake will automatically download Kokkos and HeFFTe if not found on the system. Use `Kokkos_VERSION=git.<tag>` and `Heffte_VERSION=git.<tag>` to force specific versions.
