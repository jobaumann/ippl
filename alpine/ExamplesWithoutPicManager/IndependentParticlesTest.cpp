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
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");

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

        msg << "Particles created and initial conditions assigned" << endl;

        // Initialize time and load balance frequency
        P->time_m = 0.0;
        P->loadbalancefreq_m = std::atoi(argv[arg++]);

        // Parse --overallocate parameter (currently unused, reserved for future)
        double overalloc_factor = 1.0;
        if (arg < argc && std::string(argv[arg]) == "--overallocate") {
            ++arg;
            if (arg < argc) {
                overalloc_factor = std::atof(argv[arg++]);
            }
        }
        msg << "Over-allocation factor: " << overalloc_factor << " (currently unused)" << endl;

        // get views for particle attributes
        auto Pview = P->P.getView();
        auto Qview = P->q.getView();
        auto Rview = P->R.getView();

        double B = 0.001; // magnetic field strength in z direction

        // begin main timestep loop
        msg << "Starting iterations ..." << endl;

        // =====================================================================
        // TIME-BLOCKING APPROACH WITH PARTICLE BIRTH/DEATH
        // =====================================================================
        // Divide simulation into time blocks for efficient birth/death handling
        //
        // Each block has 3 stages:
        //   1. Simulate all existing particles, track deaths and birth requests
        //   2. Destroy dead particles, create new ones, catch up new particles
        //   3. Load balance particles across MPI ranks
        //
        // This avoids dynamic particle creation mid-timestep which would
        // break the data-parallel loop structure.
        // =====================================================================

        const unsigned int time_block_size = 20;  // Timesteps per block
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

            // ============================================================
            // STAGE 1: Simulate and Track
            // ============================================================
            IpplTimings::startTimer(stage1Timer);

            size_type current_local_num = P->getLocalNum();

            // Refresh views (particle container may have resized)
            Pview = P->P.getView();
            Qview = P->q.getView();
            Rview = P->R.getView();

            // Tracking arrays for deaths and birth requests
            using bool_type = Kokkos::View<bool*>;
            using uint_type = Kokkos::View<unsigned int*>;

            bool_type died_mask("died_particles", current_local_num);
            bool_type birth_requested("birth_requested", current_local_num);
            uint_type birth_times("birth_times", current_local_num);

            // Stage 1: Simulate and track
            Kokkos::parallel_for(
                "Stage1_SimulateAndTrack",
                current_local_num,
                KOKKOS_LAMBDA(const size_type i) {
                    auto rand_gen = rand_pool_bd.get_state();

                    bool has_died = false;

                    // Simulate particle for this time block
                    for (unsigned int step = 0; step < block_steps; ++step) {
                        if (!has_died) {
                            // LeapFrog integration: https://en.wikipedia.org/wiki/Leapfrog_integration
                            // Lorentz force from constant B field: F = q(v × B)
                            double vx = Pview(i)[0];
                            double vy = Pview(i)[1];

                            // Half-step velocity update (kick)
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * vy;
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * vx;

                            // Full-step position update (drift)
                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            // Half-step velocity update (kick)
                            vx = Pview(i)[0];
                            vy = Pview(i)[1];
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * vy;
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * vx;

                            // Check for death
                            if (rand_gen.drand(0.0, 1.0) < death_chance) {
                                died_mask(i) = true;
                                has_died = true;
                            }
                        }

                        // Each particle requests birth once per block (at step 0)
                        if (step == 0 && rand_gen.drand(0.0, 1.0) < birth_chance) {
                            birth_requested(i) = true;
                            birth_times(i) = static_cast<unsigned int>(
                                rand_gen.drand(0.0, static_cast<double>(block_steps)));
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

            msg << "  Stage 1: " << current_local_num << " particles simulated, "
                << num_died << " died, " << num_births << " births requested" << endl;

            // ============================================================
            // STAGE 2: Birth and Death
            // ============================================================
            IpplTimings::startTimer(stage2Timer);

            // 2a. Destroy dead particles
            if (num_died > 0) {
                P->destroy(died_mask, num_died);
            }

            // 2b. Extract birth times before creating new particles
            using birth_info_type = Kokkos::View<unsigned int*>;
            birth_info_type stored_birth_times("stored_birth_times", num_births);

            if (num_births > 0) {
                // Compact birth times array using parallel_scan
                Kokkos::parallel_scan("Extract birth times", current_local_num,
                    KOKKOS_LAMBDA(const size_type i, size_type& idx, const bool final) {
                        if (birth_requested(i) && final) {
                            stored_birth_times(idx) = birth_times(i);
                        }
                        if (birth_requested(i)) idx += 1;
                    });
                Kokkos::fence();

                // 2c. Create new particles
                size_type old_local_num = P->getLocalNum();
                P->create(num_births);
                size_type new_local_num = P->getLocalNum();

                // Refresh views after particle creation
                Pview = P->P.getView();
                Qview = P->q.getView();
                Rview = P->R.getView();

                // 2d. Initialize and catch up new particles
                // New particles are appended: indices [old_local_num, new_local_num)
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

                        // Catch up: simulate from random birth time to end of block
                        unsigned int birth_time = stored_birth_times(birth_info_idx);
                        for (unsigned int step = birth_time; step < block_steps; ++step) {
                            // LeapFrog integration (no deaths during catch-up)
                            double vx = Pview(i)[0];
                            double vy = Pview(i)[1];

                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * vy;
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * vx;

                            Rview(i)[0] += dt * Pview(i)[0];
                            Rview(i)[1] += dt * Pview(i)[1];
                            Rview(i)[2] += dt * Pview(i)[2];

                            vx = Pview(i)[0];
                            vy = Pview(i)[1];
                            Pview(i)[0] += 0.5 * dt * B * Qview(i) * vy;
                            Pview(i)[1] -= 0.5 * dt * B * Qview(i) * vx;
                        }

                        rand_pool_bd.free_state(rand_gen);
                    });
                Kokkos::fence();
            }

            IpplTimings::stopTimer(stage2Timer);

            msg << "  Stage 2: " << num_died << " destroyed, "
                << num_births << " created and caught up" << endl;

            // ============================================================
            // STAGE 3: Load Balance
            // ============================================================
            IpplTimings::startTimer(stage3Timer);

            // Update particle distribution across MPI ranks
            // TODO: Replace with tree-based load balancing from reference/
            P->update();

            IpplTimings::stopTimer(stage3Timer);

            // Report global particle count
            size_type global_alive = 0;
            size_type local_alive = P->getLocalNum();
            ippl::Comm->reduce(local_alive, global_alive, 1, std::plus<size_type>());

            if (ippl::Comm->rank() == 0) {
                msg << "  Stage 3: " << global_alive << " particles globally after load balance" << endl;
            }
        }  // End time block loop

        IpplTimings::stopTimer(timeBlockTimer);

        msg << "Time-block simulation completed. All particles advanced " << nt << " timesteps." << endl;

        // Update final simulation time
        P->time_m = nt * dt;

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
