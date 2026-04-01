// Independent Particles Test
//   Usage:
//     srun ./IndependentParticlesTest
//                  <nx> [<ny>...] <Np> <Nt> <stype>
//                  <lbthres> --overallocate <ovfactor> --info 10
//     nx       = No. cell-centered points in the x-direction
//     ny...    = No. cell-centered points in the y-, z-, ...direction
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     stype    = Field solver type (FFT, CG, TG, and OPEN supported)
//     lbfreq   = Load balancing frequency i.e., Number of time steps after which particle
//                load balancing should happen
//     ovfactor = Over-allocation factor for the buffers used in the communication. Typical
//                values are 1.0, 2.0. Value 1.0 means no over-allocation.
//     Example:
//     srun ./IndependentParticlesTest 128 128 128 10000 10 FFT 10 --overallocate 1.0 --info 10
//     ./IndependentParticlesTest 32 32 32 10 100 FFT 10 --overallocate 1.0 --info 10
//
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"

constexpr unsigned Dim = 3;

const char* TestName = "IndependentParticlesTest";

template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type vals;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T start, end;

    // Initialize all members
    generate_random(view_type vals_, GeneratorPool rand_pool_, T start_, T end_)
        : vals(vals_)
        , rand_pool(rand_pool_)
        , start(start_)
        , end(end_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // Draw samples numbers from the pool as double in the range [start, end)
        for (unsigned d = 0; d < Dim; ++d) {
            vals(i)[d] = rand_gen.drand(start[d], end[d]);
        }

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        setSignalHandler();

        Inform msg("IndependentParticlesTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();
        int arg    = 1;

        Vector_t<int, Dim> nr;
        for (unsigned d = 0; d < Dim; d++)
            nr[d] = std::atoi(argv[arg++]);

        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef dumpDataTimer    = IpplTimings::getTimer("dumpData");
        // static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        // static IpplTimings::TimerRef temp             = IpplTimings::getTimer("randomMove");
        // static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
        // static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

        IpplTimings::startTimer(mainTimer);

        const size_type totalP = std::atoll(argv[arg++]);
        const unsigned int nt  = std::atoi(argv[arg++]);

        // over-allocation factor for birth/death - allocate 50% more particles
        const double overalloc_factor = 1.0;
        const size_type totalP_allocated = static_cast<size_type>(totalP * overalloc_factor);

        msg << "Independent Particles Test" << endl
            << "nt " << nt << " Np= " << totalP << " (allocated: " << totalP_allocated << ")"
            << " grid = " << nr << endl;

        using bunch_type = ChargedParticles<PLayout_t<double, Dim>, double, Dim>;

        std::unique_ptr<bunch_type> P;

        ippl::NDIndex<Dim> domain;
        for (unsigned i = 0; i < Dim; i++) {
            domain[i] = ippl::Index(nr[i]);
        }

        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(20.0);

        Vector_t<double, Dim> hr     = (rmax-rmin) / nr;
        Vector_t<double, Dim> origin = rmin;
        const double dt              = 1.0;

        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        const bool isAllPeriodic = true;
        Mesh_t<Dim> mesh(domain, hr, origin);
        FieldLayout_t<Dim> FL(MPI_COMM_WORLD, domain, isParallel, isAllPeriodic);
        PLayout_t<double, Dim> PL(FL, mesh);

        // Why this specific number?
        double Q           = -1562.5;
        std::string solver = argv[arg++];
        P = std::make_unique<bunch_type>(PL, hr, rmin, rmax, isParallel, Q, solver);

        P->nr_m        = nr;
        // Distribute allocated particles across ranks
        size_type nloc = totalP_allocated / ippl::Comm->size();

        int rest = (int)(totalP_allocated - nloc * ippl::Comm->size());

        if (ippl::Comm->rank() < rest)
            ++nloc;

        // Calculate how many should be initially active on this rank
        size_type nloc_active = totalP / ippl::Comm->size();
        int rest_active = (int)(totalP - nloc_active * ippl::Comm->size());
        if (ippl::Comm->rank() < rest_active)
            ++nloc_active;

        IpplTimings::startTimer(particleCreation);
        P->create(nloc);

        const ippl::NDIndex<Dim>& lDom = FL.getLocalNDIndex();
        Vector_t<double, Dim> Rmin, Rmax;
        for (unsigned d = 0; d < Dim; ++d) {
            Rmin[d] = origin[d] + lDom[d].first() * hr[d];
            Rmax[d] = origin[d] + (lDom[d].last() + 1) * hr[d];
        }

        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        // Initialize positions for all particles (active and dormant)
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool64, Rmin, Rmax));
        Kokkos::fence();

        // Initialize velocities for all particles (active and dormant)
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->P.getView(), rand_pool64, -1, 1));
        Kokkos::fence();

        // Set charge: active particles get proper charge, dormant particles get q=0
        {
            auto Qview_init = P->q.getView();
            const double active_charge_init = P->Q_m / totalP;
            Kokkos::parallel_for(
                nloc, KOKKOS_LAMBDA(const size_type i) {
                    if (i < nloc_active) {
                        Qview_init(i) = active_charge_init;  // Active particle
                    } else {
                        Qview_init(i) = 0.0;  // Dormant particle
                    }
                });
            Kokkos::fence();
        }


        IpplTimings::stopTimer(particleCreation);

        P->initializeFields(mesh, FL);

        IpplTimings::startTimer(updateTimer);
        P->update();
        IpplTimings::stopTimer(updateTimer);

        msg << "particles created and initial conditions assigned " << endl;

        P->initSolver();
        P->time_m            = 0.0;
        P->loadbalancefreq_m = std::atoi(argv[arg++]);

        // Optional: birth/death rates from CLI (default 0.0001)
        double death_chance_param = 0.0001;
        double birth_chance_param = 0.0001;
        // Parse --overallocate first, then check for birth/death after --info
        // Look for --death and --birth flags anywhere in argv
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--death" && i + 1 < argc)
                death_chance_param = std::atof(argv[i + 1]);
            if (std::string(argv[i]) == "--birth" && i + 1 < argc)
                birth_chance_param = std::atof(argv[i + 1]);
        }

        IpplTimings::startTimer(DummySolveTimer);
        P->rho_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP_allocated, 0, hr);
        P->initializeORB(FL, mesh);
        // bool fromAnalyticDensity = false;

        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        P->gatherCIC();

        IpplTimings::startTimer(dumpDataTimer);
        P->dumpData();
        P->gatherStatistics(totalP_allocated);
        IpplTimings::stopTimer(dumpDataTimer);

        // get views for particle attributes
        auto Pview = P->P.getView();
        auto Qview = P->q.getView();
        auto Rview = P->R.getView();

        double B = 0.001; // magnetic field strength in z direction
        const double kick = 0.5 * dt * B * (P->Q_m / totalP);

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;

        // Task-parallel approach: Each particle executes all timesteps independently
        // NOTE: This approach only works for independent particles (no field solve, no boundary crossing)

        // Set checkpoint frequency for intermediate output (0 = no intermediate output)
        // For intermediate VTK files and statistics, set this to a positive value
        const unsigned int checkpointFreq = 0;  // Set to 0 for pure task-parallel (fastest)
                                                 // Set to >0 for intermediate output every N steps

        static IpplTimings::TimerRef taskParallelTimer = IpplTimings::getTimer("taskParallelLoop");
        IpplTimings::startTimer(taskParallelTimer);

        double death_chance = death_chance_param;
        double birth_chance = birth_chance_param;
        msg << "death_chance= " << death_chance << " birth_chance= " << birth_chance << endl;

        // Variables needed for birth (captured by lambda)
        const double active_charge = P->Q_m / totalP;
        const Vector_t<double, Dim> birth_rmin = rmin;
        const Vector_t<double, Dim> birth_rmax = rmax;

        // Create random pool for birth/death (reuse the one from initialization with different seed)
        Kokkos::Random_XorShift64_Pool<> rand_pool_bd((size_type)(1337 + 100 * ippl::Comm->rank()));

        const bool has_birth_death = (death_chance > 0.0 || birth_chance > 0.0);

        if (checkpointFreq == 0 && !has_birth_death) {
            // Pure physics: no birth/death, no RNG overhead
            Kokkos::parallel_for(
                "IndependentParticleLoop_NoBD",
                P->getLocalNum(),
                KOKKOS_LAMBDA(const size_type i) {
                    double vx = Pview(i)[0], vy = Pview(i)[1], vz = Pview(i)[2];
                    double rx = Rview(i)[0], ry = Rview(i)[1], rz = Rview(i)[2];
                    for (unsigned int it = 0; it < nt; it++) {
                        vx += kick * vy;
                        vy -= kick * vx;

                        rx += dt * vx;
                        ry += dt * vy;
                        rz += dt * vz;

                        vx += kick * vy;
                        vy -= kick * vx;
                    }
                    Pview(i)[0] = vx; Pview(i)[1] = vy; Pview(i)[2] = vz;
                    Rview(i)[0] = rx; Rview(i)[1] = ry; Rview(i)[2] = rz;
                }
            );
            Kokkos::fence();
        } else if (checkpointFreq == 0) {
            // Pure task-parallel with birth/death
            Kokkos::parallel_for(
                "IndependentParticleLoop",
                P->getLocalNum(),
                KOKKOS_LAMBDA(const size_type i) {
                    auto rand_gen = rand_pool_bd.get_state();

                    for (unsigned int it = 0; it < nt; it++) {
                        bool is_active = (Qview(i) != 0.0);

                        if (is_active) {
                            Pview(i)[0] += kick * Pview(i)[1];
                            Pview(i)[1] -= kick * Pview(i)[0];

                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            Pview(i)[0] += kick * Pview(i)[1];
                            Pview(i)[1] -= kick * Pview(i)[0];

                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < death_chance) {
                                Qview(i) = 0.0;
                            }
                        } else {
                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < birth_chance) {
                                Qview(i) = active_charge;

                                // Initialize new particle at random position in domain
                                for (unsigned d = 0; d < Dim; ++d) {
                                    Rview(i)[d] = rand_gen.drand(birth_rmin[d], birth_rmax[d]);
                                    Pview(i)[d] = rand_gen.drand(-1.0, 1.0);
                                }
                            }
                        }
                    }

                    // Return the generator state to the pool
                    rand_pool_bd.free_state(rand_gen);
                }
            );
            Kokkos::fence();
        } else {
            // Hybrid approach: Break into chunks for intermediate output
            msg << "Using hybrid task-parallel with checkpoints every " << checkpointFreq << " steps" << endl;
            for (unsigned int chunk_start = 0; chunk_start < nt; chunk_start += checkpointFreq) {
                unsigned int chunk_end = std::min(chunk_start + checkpointFreq, nt);
                unsigned int chunk_size = chunk_end - chunk_start;

                // Run task-parallel chunk
                Kokkos::parallel_for(
                    "IndependentParticleChunk",
                    P->getLocalNum(),
                    KOKKOS_LAMBDA(const size_type i) {
                        // Get random number generator state from pool for this particle
                        auto rand_gen = rand_pool_bd.get_state();

                        for (unsigned int it = 0; it < chunk_size; it++) {
                            bool is_active = (Qview(i) != 0.0);

                            if (is_active) {
                                // Active particle: run physics
                                // kick
                                Pview(i)[0] += kick * Pview(i)[1];
                                Pview(i)[1] -= kick * Pview(i)[0];

                                // drift
                                Rview(i)[0] += dt * Pview(i)[0];
                                Rview(i)[1] += dt * Pview(i)[1];
                                Rview(i)[2] += dt * Pview(i)[2];

                                // kick
                                Pview(i)[0] += kick * Pview(i)[1];
                                Pview(i)[1] -= kick * Pview(i)[0];

                                // Death check
                                double rand_val = rand_gen.drand(0.0, 1.0);
                                if (rand_val < death_chance) {
                                    Qview(i) = 0.0;  // Mark particle as dead
                                }
                            } else {
                                // Dormant particle: check for birth
                                double rand_val = rand_gen.drand(0.0, 1.0);
                                if (rand_val < birth_chance) {
                                    // Birth: activate dormant particle
                                    Qview(i) = active_charge;

                                    // Initialize new particle at random position
                                    for (unsigned d = 0; d < Dim; ++d) {
                                        Rview(i)[d] = rand_gen.drand(birth_rmin[d], birth_rmax[d]);
                                        Pview(i)[d] = rand_gen.drand(-1.0, 1.0);
                                    }
                                }
                            }
                        }

                        // Return the generator state to the pool
                        rand_pool_bd.free_state(rand_gen);
                    }
                );
                Kokkos::fence();

                // Intermediate output at checkpoint
                P->time_m = chunk_end * dt;
                P->scatterCIC(totalP_allocated, chunk_end, hr);

                // Uncomment to enable intermediate VTK files:
                dumpVTK(P->rho_m, P->nr_m[0], P->nr_m[1], P->nr_m[2], chunk_end,
                        P->hr_m[0], P->hr_m[1], P->hr_m[2]);

                P->dumpData();
                P->gatherStatistics(totalP_allocated);

                msg << "Checkpoint: completed timestep " << chunk_end << " / " << nt << endl;
            }
        }

        IpplTimings::stopTimer(taskParallelTimer);

        // Update final simulation time
        P->time_m = nt * dt;

        msg << "Task-parallel loop completed. All particles advanced " << nt << " timesteps." << endl;

        // Final output
        // Note: For pure performance testing, we skip detailed output
        // Uncomment below for full diagnostics (requires particle update first)

        /*
        // Update particles to correct MPI ranks after movement
        IpplTimings::startTimer(updateTimer);
        P->update();
        IpplTimings::stopTimer(updateTimer);

        IpplTimings::startTimer(dumpDataTimer);
        P->scatterCIC(totalP_allocated, nt, hr);

        // Optional: Generate final VTK file for visualization
        // dumpVTK(P->rho_m, P->nr_m[0], P->nr_m[1], P->nr_m[2], nt, P->hr_m[0], P->hr_m[1], P->hr_m[2]);

        // Dump final statistics
        P->dumpData();
        P->gatherStatistics(totalP_allocated);
        IpplTimings::stopTimer(dumpDataTimer);
        */

        msg << "Skipping detailed final output for performance test" << endl;

        msg << "Independent Particles Test: End." << endl;
        IpplTimings::stopTimer(mainTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;
    }
    ippl::finalize();

    return 0;
}
