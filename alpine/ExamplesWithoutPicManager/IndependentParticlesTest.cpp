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
        static IpplTimings::TimerRef PTimer           = IpplTimings::getTimer("pushVelocity");
        static IpplTimings::TimerRef temp             = IpplTimings::getTimer("randomMove");
        static IpplTimings::TimerRef RTimer           = IpplTimings::getTimer("pushPosition");
        static IpplTimings::TimerRef updateTimer      = IpplTimings::getTimer("update");
        static IpplTimings::TimerRef DummySolveTimer  = IpplTimings::getTimer("solveWarmup");
        static IpplTimings::TimerRef SolveTimer       = IpplTimings::getTimer("solve");
        static IpplTimings::TimerRef domainDecomposition = IpplTimings::getTimer("loadBalance");

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
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool64, Rmin, Rmax));
        Kokkos::fence();
        P->q = P->Q_m / totalP;

        // Initialize random particle velocities
        // P->P = 0.0;
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->P.getView(), rand_pool64, -1, 1));
        Kokkos::fence();


        IpplTimings::stopTimer(particleCreation);

        P->initializeFields(mesh, FL);

        IpplTimings::startTimer(updateTimer);
        P->update();
        IpplTimings::stopTimer(updateTimer);

        msg << "particles created and initial conditions assigned " << endl;

        P->initSolver();
        P->time_m            = 0.0;
        P->loadbalancefreq_m = std::atoi(argv[arg++]);

        IpplTimings::startTimer(DummySolveTimer);
        P->rho_m = 0.0;
        P->runSolver();
        IpplTimings::stopTimer(DummySolveTimer);

        P->scatterCIC(totalP, 0, hr);
        P->initializeORB(FL, mesh);
        bool fromAnalyticDensity = false;

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

        // Task-parallel approach: Each particle executes all timesteps independently
        // NOTE: This approach only works for independent particles (no field solve, no boundary crossing)

        // Set checkpoint frequency for intermediate output (0 = no intermediate output)
        // For intermediate VTK files and statistics, set this to a positive value
        const unsigned int checkpointFreq = 0;  // Set to 0 for pure task-parallel (fastest)
                                                 // Set to >0 for intermediate output every N steps

        static IpplTimings::TimerRef taskParallelTimer = IpplTimings::getTimer("taskParallelLoop");
        IpplTimings::startTimer(taskParallelTimer);

        if (checkpointFreq == 0) {
            // Pure task-parallel: Run all timesteps in one kernel (fastest)
            Kokkos::parallel_for(
                "IndependentParticleLoop",
                P->getLocalNum(),
                KOKKOS_LAMBDA(const size_type i) {
                    // Each particle independently executes all timesteps
                    for (unsigned int it = 0; it < nt; it++) {
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
                    }
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
                        for (unsigned int it = 0; it < chunk_size; it++) {
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
                        }
                    }
                );
                Kokkos::fence();

                // Since the particles have moved spatially update them to correct processors
                IpplTimings::startTimer(updateTimer);
                P->update();
                IpplTimings::stopTimer(updateTimer);

                // Domain Decomposition
                if (P->balance(totalP, chunk_end)) {
                    msg << "Starting repartition" << endl;
                    IpplTimings::startTimer(domainDecomposition);
                    P->repartition(FL, mesh, fromAnalyticDensity);
                    IpplTimings::stopTimer(domainDecomposition);
                }

                // Intermediate output at checkpoint
                P->time_m = chunk_end * dt;
                P->scatterCIC(totalP, chunk_end, hr);

                // Uncomment to enable intermediate VTK files:
                dumpVTK(P->rho_m, P->nr_m[0], P->nr_m[1], P->nr_m[2], chunk_end,
                        P->hr_m[0], P->hr_m[1], P->hr_m[2]);

                P->dumpData();
                P->gatherStatistics(totalP);

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
        P->scatterCIC(totalP, nt, hr);

        // Optional: Generate final VTK file for visualization
        // dumpVTK(P->rho_m, P->nr_m[0], P->nr_m[1], P->nr_m[2], nt, P->hr_m[0], P->hr_m[1], P->hr_m[2]);

        // Dump final statistics
        P->dumpData();
        P->gatherStatistics(totalP);
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
