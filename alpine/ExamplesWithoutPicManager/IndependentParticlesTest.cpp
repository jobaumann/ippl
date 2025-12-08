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

        msg << "Independent Particles Test" << endl
            << "nt " << nt << " Np= " << totalP << " grid = " << nr << endl;

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
        size_type nloc = totalP / ippl::Comm->size();

        int rest = (int)(totalP - nloc * ippl::Comm->size());

        if (ippl::Comm->rank() < rest)
            ++nloc;

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

        // Initialize velocities for all particles
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->P.getView(), rand_pool64, -1, 1));
        Kokkos::fence();

        // Set charge for all particles
        P->q = P->Q_m / totalP;


        IpplTimings::stopTimer(particleCreation);

        P->initializeFields(mesh, FL);

        IpplTimings::startTimer(updateTimer);
        P->update();
        IpplTimings::stopTimer(updateTimer);

        msg << "particles created and initial conditions assigned " << endl;

        P->initSolver();
        P->time_m            = 0.0;
        P->loadbalancefreq_m = std::atoi(argv[arg++]);

        // Parse --overallocate parameter
        double overalloc_factor = 1.0;  // Default: no over-allocation
        if (arg < argc && std::string(argv[arg]) == "--overallocate") {
            ++arg;
            if (arg < argc) {
                overalloc_factor = std::atof(argv[arg++]);
            }
        }
        msg << "Over-allocation factor: " << overalloc_factor << endl;

        IpplTimings::startTimer(DummySolveTimer);
        P->rho_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP, 0, hr);
        P->initializeORB(FL, mesh);
        // bool fromAnalyticDensity = false;

        IpplTimings::startTimer(SolveTimer);
        P->runSolver();
        IpplTimings::stopTimer(SolveTimer);

        P->gatherCIC();

        IpplTimings::startTimer(dumpDataTimer);
        P->dumpData();
        P->gatherStatistics(totalP);
        IpplTimings::stopTimer(dumpDataTimer);

        // get views for particle attributes
        auto Pview = P->P.getView();
        auto Qview = P->q.getView();
        auto Rview = P->R.getView();

        double B = 0.001; // magnetic field strength in z direction

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;

        // 3-Stage Time-Blocking approach with Birth/Death and Load Balancing
        // Stage 1: Simulate existing particles, track deaths and count births needed
        // Stage 2: Destroy dead particles, create new particles, catch up new particles
        // Stage 3: Load balance alive particles across ranks

        const unsigned int time_block_size = 20;  // Number of timesteps per time block
        const unsigned int num_time_blocks = (nt + time_block_size - 1) / time_block_size;

        static IpplTimings::TimerRef timeBlockTimer = IpplTimings::getTimer("timeBlockLoop");
        static IpplTimings::TimerRef stage1Timer = IpplTimings::getTimer("stage1Simulation");
        static IpplTimings::TimerRef stage2Timer = IpplTimings::getTimer("stage2BirthDeath");
        static IpplTimings::TimerRef stage3Timer = IpplTimings::getTimer("stage3LoadBalance");

        IpplTimings::startTimer(timeBlockTimer);

        double death_chance = 0.0001;  // Probability per timestep for particle to die
        double birth_chance = 0.0001;  // Probability per timestep per particle for birth

        // Variables needed for birth (captured by lambda)
        const double active_charge = P->Q_m / totalP;
        const Vector_t<double, Dim> birth_rmin = rmin;
        const Vector_t<double, Dim> birth_rmax = rmax;

        // Create random pool for birth/death
        Kokkos::Random_XorShift64_Pool<> rand_pool_bd((size_type)(1337 + 100 * ippl::Comm->rank()));

        // Time block loop
        for (unsigned int block = 0; block < num_time_blocks; ++block) {
            unsigned int block_start = block * time_block_size;
            unsigned int block_end = std::min(block_start + time_block_size, nt);
            unsigned int block_steps = block_end - block_start;

            msg << "Time block " << (block + 1) << "/" << num_time_blocks
                << " (timesteps " << block_start << "-" << block_end << ")" << endl;

            // =================================================================
            // STAGE 1: Simulate existing particles, track deaths and births
            // =================================================================
            IpplTimings::startTimer(stage1Timer);

            size_type current_local_num = P->getLocalNum();

            // Get fresh views (may have changed from previous block)
            Pview = P->P.getView();
            Qview = P->q.getView();
            Rview = P->R.getView();

            // Tracking structures
            using bool_type = Kokkos::View<bool*>;
            bool_type died_mask("died_particles", current_local_num);

            // Count births per particle (each particle can trigger 0 or 1 birth per block)
            using uint_type = Kokkos::View<unsigned int*>;
            uint_type birth_times("birth_times", current_local_num);
            bool_type birth_requested("birth_requested", current_local_num);

            // Stage 1: Simulate and track
            Kokkos::parallel_for(
                "Stage1_SimulateAndTrack",
                current_local_num,
                KOKKOS_LAMBDA(const size_type i) {
                    auto rand_gen = rand_pool_bd.get_state();

                    bool has_died = false;

                    // Simulate for this time block
                    for (unsigned int step = 0; step < block_steps; ++step) {
                        if (!has_died) {
                            // Active particle: run physics (LeapFrog)
                            // kick (first half)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                            // drift
                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            // kick (second half)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                            // Death check
                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < death_chance) {
                                died_mask(i) = true;
                                has_died = true;
                            }
                        }

                        // Each particle can request a birth once per block
                        if (step == 0) {
                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < birth_chance) {
                                birth_requested(i) = true;
                                // Random birth time within this block
                                birth_times(i) = static_cast<unsigned int>(
                                    rand_gen.drand(0.0, static_cast<double>(block_steps)));
                            }
                        }
                    }

                    rand_pool_bd.free_state(rand_gen);
                });
            Kokkos::fence();

            // Count deaths and birth requests
            size_type num_died = 0;
            size_type num_births = 0;

            Kokkos::parallel_reduce("Count deaths", current_local_num,
                KOKKOS_LAMBDA(const size_type i, size_type& sum) {
                    if (died_mask(i)) sum += 1;
                }, num_died);

            Kokkos::parallel_reduce("Count births", current_local_num,
                KOKKOS_LAMBDA(const size_type i, size_type& sum) {
                    if (birth_requested(i)) sum += 1;
                }, num_births);

            IpplTimings::stopTimer(stage1Timer);

            msg << "  Stage 1: Simulated " << current_local_num << " particles, "
                << num_died << " deaths, " << num_births << " birth requests" << endl;

            // =================================================================
            // STAGE 2: Destroy dead particles, create new, catch up new particles
            // =================================================================
            IpplTimings::startTimer(stage2Timer);

            // Destroy dead particles
            if (num_died > 0) {
                P->destroy(died_mask, num_died);
                msg << "  Stage 2a: Destroyed " << num_died << " particles" << endl;
            }

            // Store birth times before creating (need to extract from old indices)
            using birth_info_type = Kokkos::View<unsigned int*>;
            birth_info_type stored_birth_times("stored_birth_times", num_births);

            if (num_births > 0) {
                // Extract birth times for particles that requested birth
                size_type birth_idx = 0;
                Kokkos::parallel_scan("Extract birth times", current_local_num,
                    KOKKOS_LAMBDA(const size_type i, size_type& idx, const bool final) {
                        if (birth_requested(i)) {
                            if (final) {
                                stored_birth_times(idx) = birth_times(i);
                            }
                            idx += 1;
                        }
                    }, birth_idx);
                Kokkos::fence();

                // Create new particles
                size_type old_local_num = P->getLocalNum();
                P->create(num_births);

                msg << "  Stage 2b: Created " << num_births << " new particles" << endl;

                // Get updated views after creation
                Pview = P->P.getView();
                Qview = P->q.getView();
                Rview = P->R.getView();
                size_type new_local_num = P->getLocalNum();

                // Initialize and catch up newly created particles
                // New particles are at indices [old_local_num, new_local_num)
                Kokkos::parallel_for(
                    "Stage2_InitializeAndCatchUp",
                    Kokkos::RangePolicy<>(old_local_num, new_local_num),
                    KOKKOS_LAMBDA(const size_type i) {
                        auto rand_gen = rand_pool_bd.get_state();

                        size_type birth_info_idx = i - old_local_num;

                        // Initialize particle at random position and velocity
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] = rand_gen.drand(birth_rmin[d], birth_rmax[d]);
                            Pview(i)[d] = rand_gen.drand(-1.0, 1.0);
                        }

                        // Set charge
                        Qview(i) = active_charge;

                        // Catch up: simulate from birth time to end of block
                        unsigned int birth_time = stored_birth_times(birth_info_idx);
                        for (unsigned int step = birth_time; step < block_steps; ++step) {
                            // LeapFrog integration (no births/deaths during catch-up)
                            // kick (first half)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                            // drift
                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            // kick (second half)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];
                        }

                        rand_pool_bd.free_state(rand_gen);
                    });
                Kokkos::fence();

                msg << "  Stage 2c: Caught up " << num_births << " new particles" << endl;
            }

            IpplTimings::stopTimer(stage2Timer);

            // =================================================================
            // STAGE 3: Load balance particles across ranks
            // =================================================================
            IpplTimings::startTimer(stage3Timer);

            // TODO: Implement tree-based load balancing here
            // For now, just update particle positions across ranks
            P->update();

            size_type global_alive = 0;
            size_type local_alive = P->getLocalNum();
            ippl::Comm->reduce(local_alive, global_alive, 1, std::plus<size_type>());

            IpplTimings::stopTimer(stage3Timer);

            if (ippl::Comm->rank() == 0) {
                msg << "  Stage 3: Load balanced, " << global_alive << " particles alive globally" << endl;
            }
        }  // End time block loop

        IpplTimings::stopTimer(timeBlockTimer);

        msg << "Time-block simulation completed" << endl;

        // Update final simulation time
        P->time_m = nt * dt;

        // Skip the old implementation (kept for reference)
        if (false) {
            // OLD CODE: Pure task-parallel: Run all timesteps in one kernel (fastest)
            Kokkos::parallel_for(
                "IndependentParticleLoop",
                P->getLocalNum(),
                KOKKOS_LAMBDA(const size_type i) {
                    // Get random number generator state from pool for this particle
                    auto rand_gen = rand_pool_bd.get_state();

                    // Each particle independently executes all timesteps
                    for (unsigned int it = 0; it < nt; it++) {
                        bool is_active = (Qview(i) != 0.0);

                        if (is_active) {
                            // Active particle: run physics
                            // LeapFrog time stepping https://en.wikipedia.org/wiki/Leapfrog_integration
                            // Here, we assume a constant charge-to-mass ratio of -1 for
                            // all the particles hence eliminating the need to store mass as
                            // an attribute

                            // kick (first half of velocity update)
                            // Constant magnetic field to test independent particle motion
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                            // drift (position update)
                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            // kick (second half of velocity update)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                            // Death check: random particle death for more complex workload balancing
                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < death_chance) {
                                Qview(i) = 0.0;  // Mark particle as dead (dormant)
                            }
                        } else {
                            // Dormant particle: check for birth
                            double rand_val = rand_gen.drand(0.0, 1.0);
                            if (rand_val < birth_chance) {
                                // Birth: activate dormant particle
                                Qview(i) = active_charge;  // Set proper charge

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
                                Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                                Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

                                // drift
                                Rview(i)[0] += dt * Pview(i)[0];
                                Rview(i)[1] += dt * Pview(i)[1];
                                Rview(i)[2] += dt * Pview(i)[2];

                                // kick
                                Pview(i)[0] += 0.5 * dt * B * Qview(i) * Pview(i)[1];
                                Pview(i)[1] -= 0.5 * dt * B * Qview(i) * Pview(i)[0];

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
        }  // End if (false) - old code

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
