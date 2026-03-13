// Independent Particles Test with Birth/Death and Tree-Based Load Balancing
//
//   Usage:
//     srun ./IndependentParticlesTest <Np> <Nt> <lbfreq> --info 10
//
//     Np       = Total no. of macro-particles in the simulation
//     Nt       = Number of time steps
//     lbfreq   = Load balancing frequency (timesteps per time block)
//
//   Example:
//     ./IndependentParticlesTest 1000 100 20 --info 10
//
#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>

#include "Utility/IpplTimings.h"
#include "ParticleTreeLayout.hpp"

constexpr unsigned Dim = 3;

const char* TestName = "IndependentParticlesTest";

// Functor to generate random vectors
template <typename T, class GeneratorPool, unsigned Dim>
struct generate_random {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;

    view_type vals;
    GeneratorPool rand_pool;
    T start, end;

    generate_random(view_type vals_, GeneratorPool rand_pool_, T start_, T end_)
        : vals(vals_)
        , rand_pool(rand_pool_)
        , start(start_)
        , end(end_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();
        for (unsigned d = 0; d < Dim; ++d) {
            vals(i)[d] = rand_gen.drand(start[d], end[d]);
        }
        rand_pool.free_state(rand_gen);
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg("IndependentParticlesTest");
        auto start = std::chrono::high_resolution_clock::now();

        // Parse command line arguments
        int arg = 1;
        const size_type totalP    = std::atoll(argv[arg++]);
        const unsigned int nt     = std::atoi(argv[arg++]);
        const unsigned int lbfreq = std::atoi(argv[arg++]);

        // Check for --debug flag
        bool debug = false;
        for (int i = 1; i < argc; ++i) {
            if (std::string(argv[i]) == "--debug") {
                debug = true;
                break;
            }
        }

        msg << "Independent Particles Test" << endl
            << "Np= " << totalP << " Nt= " << nt << " Load balance freq= " << lbfreq << endl;
        if (debug) {
            msg << "Debug mode: ENABLED" << endl;
        }

        // Simulation parameters
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(20.0);
        const double dt = 1.0;
        const double B  = 0.001;  // Magnetic field strength in z direction
        const double Q  = -1562.5;  // Total charge

        double death_chance = 0.001;  // Probability per timestep for particle to die
        double birth_chance = 0.001;  // Probability per timestep per particle for birth

        // Create particle container with tree-based layout
        using bunch_type = IndependentParticles<ParticleTreeLayout<double, Dim>, double, Dim>;
        ParticleTreeLayout<double, Dim> PL(debug);
        std::unique_ptr<bunch_type> P = std::make_unique<bunch_type>(PL);

        // Create particles with deliberate imbalance for testing
        // Rank 0 gets 50% of particles, others share the rest
        size_type nloc;
        if (ippl::Comm->rank() == 0) {
            nloc = totalP / 2;  // Rank 0 gets half
        } else {
            // Other ranks share the remaining half
            size_type remaining = totalP - (totalP / 2);
            nloc = remaining / (ippl::Comm->size() - 1);
            int rest = remaining % (ippl::Comm->size() - 1);
            if (ippl::Comm->rank() - 1 < rest) {
                ++nloc;
            }
        }

        if (debug) {
            msg << "Rank " << ippl::Comm->rank() << " creating " << nloc
                << " particles (imbalanced)" << endl;
        }

        // Timers
        static IpplTimings::TimerRef mainTimer        = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef particleCreation = IpplTimings::getTimer("particlesCreation");
        static IpplTimings::TimerRef timeBlockTimer   = IpplTimings::getTimer("timeBlockLoop");
        static IpplTimings::TimerRef stage1Timer      = IpplTimings::getTimer("stage1Simulation");
        static IpplTimings::TimerRef stage2Timer      = IpplTimings::getTimer("stage2BirthDeath");
        static IpplTimings::TimerRef stage3Timer      = IpplTimings::getTimer("stage3LoadBalance");

        IpplTimings::startTimer(mainTimer);
        IpplTimings::startTimer(particleCreation);

        P->create(nloc);

        // Initialize particle positions randomly in domain
        Kokkos::Random_XorShift64_Pool<> rand_pool_init(
            (size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool_init, rmin, rmax));
        Kokkos::fence();

        // Initialize particle velocities
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->P.getView(), rand_pool_init, -1.0, 1.0));
        Kokkos::fence();

        // Set charge for all particles
        const double charge_per_particle = Q / totalP;
        P->q                             = charge_per_particle;

        IpplTimings::stopTimer(particleCreation);

        msg << "Particles created and initial conditions assigned" << endl;

        // Get views for particle attributes
        auto Pview = P->P.getView();
        auto Qview = P->q.getView();
        auto Rview = P->R.getView();

        // Random pool for birth/death
        Kokkos::Random_XorShift64_Pool<> rand_pool_bd(
            (size_type)(1337 + 100 * ippl::Comm->rank()));

        // =====================================================================
        // TIME-BLOCKING APPROACH WITH PARTICLE BIRTH/DEATH
        // =====================================================================
        // Divide simulation into time blocks for efficient birth/death handling
        //
        // Each block has 3 stages:
        //   1. Simulate all existing particles, track deaths and birth requests
        //   2. Destroy dead particles, create new ones, catch up new particles
        //   3. Load balance particles across MPI ranks using tree-based algorithm
        // =====================================================================

        const unsigned int time_block_size = lbfreq;
        const unsigned int num_time_blocks = (nt + time_block_size - 1) / time_block_size;

        msg << "Starting iterations ..." << endl;
        IpplTimings::startTimer(timeBlockTimer);

        for (unsigned int block = 0; block < num_time_blocks; ++block) {
            unsigned int block_start = block * time_block_size;
            unsigned int block_end   = std::min(block_start + time_block_size, nt);
            unsigned int block_steps = block_end - block_start;

            msg << "Time block " << (block + 1) << "/" << num_time_blocks << " (timesteps "
                << block_start << "-" << block_end << ")" << endl;

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

            // Simulate and track births/deaths
            Kokkos::parallel_for(
                "Stage1_SimulateAndTrack", current_local_num,
                KOKKOS_LAMBDA(const size_type i) {
                    auto rand_gen = rand_pool_bd.get_state();
                    bool has_died = false;

                    // Simulate particle for this time block
                    for (unsigned int step = 0; step < block_steps; ++step) {
                        if (!has_died) {
                            // LeapFrog integration with Lorentz force: F = q(v × B)
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

                            // Check for death at each timestep
                            if (rand_gen.drand(0.0, 1.0) < death_chance) {
                                died_mask(i) = true;
                                has_died     = true;
                            }

                            // Check for birth at each timestep
                            if (!birth_requested(i) && rand_gen.drand(0.0, 1.0) < birth_chance) {
                                birth_requested(i) = true;
                                birth_times(i)     = step;
                            }
                        }
                    }

                    rand_pool_bd.free_state(rand_gen);
                });
            Kokkos::fence();

            // Count deaths and birth requests
            size_type num_died   = 0;
            size_type num_births = 0;

            Kokkos::parallel_reduce(
                "Count deaths", current_local_num,
                KOKKOS_LAMBDA(const size_type i, size_type& sum) {
                    if (died_mask(i)) sum += 1;
                },
                num_died);

            Kokkos::parallel_reduce(
                "Count births", current_local_num,
                KOKKOS_LAMBDA(const size_type i, size_type& sum) {
                    if (birth_requested(i)) sum += 1;
                },
                num_births);

            IpplTimings::stopTimer(stage1Timer);

            msg << "  Stage 1: " << current_local_num << " particles simulated, " << num_died
                << " died, " << num_births << " births requested" << endl;

            // ============================================================
            // STAGE 2: Birth and Death
            // ============================================================
            IpplTimings::startTimer(stage2Timer);

            // 2a. Destroy dead particles
            // Always call destroy (even with 0) — it contains an allreduce
            P->destroy(died_mask, num_died);

            // 2b. Extract birth times before creating new particles
            using birth_info_type = Kokkos::View<unsigned int*>;
            birth_info_type stored_birth_times("stored_birth_times", num_births);

            if (num_births > 0) {
                // Compact birth times array using parallel_scan
                Kokkos::parallel_scan(
                    "Extract birth times", current_local_num,
                    KOKKOS_LAMBDA(const size_type i, size_type& idx, const bool final) {
                        if (birth_requested(i) && final) {
                            stored_birth_times(idx) = birth_times(i);
                        }
                        if (birth_requested(i)) idx += 1;
                    });
                Kokkos::fence();
            }

            // 2c. Create new particles
            // Always call create (even with 0) — it contains an allreduce
            {
                size_type old_local_num = P->getLocalNum();
                P->create(num_births);
                size_type new_local_num = P->getLocalNum();

                // Refresh views after particle creation
                Pview = P->P.getView();
                Qview = P->q.getView();
                Rview = P->R.getView();

                // 2d. Initialize and catch up new particles
                Kokkos::parallel_for(
                    "Stage2_InitializeAndCatchUp", Kokkos::RangePolicy<>(old_local_num, new_local_num),
                    KOKKOS_LAMBDA(const size_type i) {
                        auto rand_gen            = rand_pool_bd.get_state();
                        size_type birth_info_idx = i - old_local_num;

                        // Initialize particle at random position and velocity
                        for (unsigned d = 0; d < Dim; ++d) {
                            Rview(i)[d] = rand_gen.drand(rmin[d], rmax[d]);
                            Pview(i)[d] = rand_gen.drand(-1.0, 1.0);
                        }

                        // Set charge
                        Qview(i) = charge_per_particle;

                        // Catch up: simulate from birth time to end of block
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

            msg << "  Stage 2: " << num_died << " destroyed, " << num_births
                << " created and caught up" << endl;

            // ============================================================
            // STAGE 3: Tree-Based Load Balance
            // ============================================================
            IpplTimings::startTimer(stage3Timer);

            P->getLayout().loadbalance(*P);

            IpplTimings::stopTimer(stage3Timer);

            // Report global particle count
            size_type global_alive = 0;
            size_type local_alive = P->getLocalNum();
            ippl::Comm->reduce(local_alive, global_alive, 1, std::plus<size_type>());

            if (ippl::Comm->rank() == 0) {
                msg << "  Stage 3: " << global_alive << " particles globally after load balance" << endl;
            }
        }

        IpplTimings::stopTimer(timeBlockTimer);

        msg << "Time-block simulation completed. All particles advanced " << nt << " timesteps."
            << endl;

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
