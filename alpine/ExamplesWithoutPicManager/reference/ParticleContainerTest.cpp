// Particle Container Test
// Usage:
//          srun ./ParticleContainerTest
//              <nSp> <ppsp> <bfreq> <dfreq> --overallocate 2.0 --info 5
//
//          nSp     = No. species 
//          ppsp    = No. particles per species
//          bdfreq  = birth/death frequency

#include "Ippl.h"

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "Utility/IpplTimings.h"
#include "Utility/TypeUtils.h"
#include "Types/IpplTypes.h"
#include "Particle/ParticleBase.h"
#include "Particle/ParticleLayout.h"
#include "Region/NDRegion.h"
#include "Communicate/DataTypes.h"

constexpr unsigned Dim = 3;

const char* TestName = "ParticleContainer";

// typedefs
template <unsigned Dim = 3>
using Mesh_t = ippl::UniformCartesian<double, Dim>;

template <typename T, unsigned Dim = 3>
using PLayout_t = typename ippl::ParticleSpatialLayout<T, Dim, Mesh_t<Dim>>;

template <unsigned Dim = 3>
using FieldLayout_t = ippl::FieldLayout<Dim>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector_t = ippl::Vector<T, Dim>;

using bool_type   = typename ippl::detail::ViewType<bool, 1>::view_type;

// random generator
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

// random int generator
template <typename T, class GeneratorPool>
struct generate_random_int {
    using view_type = typename ippl::detail::ViewType<T, 1>::view_type;
    // Output View for the random numbers
    view_type vals;

    // The GeneratorPool
    GeneratorPool rand_pool;

    T start, end;

    // Initialize all members
    generate_random_int(view_type vals_, GeneratorPool rand_pool_)
        : vals(vals_)
        , rand_pool(rand_pool_) {}

    KOKKOS_INLINE_FUNCTION void operator()(const size_t i) const {
        // Get a random number state from the pool for the active thread
        typename GeneratorPool::generator_type rand_gen = rand_pool.get_state();

        // Draw 0 or 1
        vals(i) = rand_gen.rand_int(2);

        // Give the state back, which will allow another thread to acquire it
        rand_pool.free_state(rand_gen);
    }
};

// Particle Layout, since we don't have Fields and don't need ParticleSpatialLayout
template <typename T, unsigned Dim, typename... PositionProperties>
class SecondaryParticleLayout : public ippl::detail::ParticleLayout<T, Dim, PositionProperties...>{
public:
    SecondaryParticleLayout() : ippl::detail::ParticleLayout<T, Dim, PositionProperties...>() {}
    ~SecondaryParticleLayout() = default;

public:
    void setDomain(ippl::NDRegion<T,Dim> domain){
        domain_m=domain;
    }
    template <class ParticleContainer>
    void update(ParticleContainer& pc){
        // Apply BC
        this->applyBC(pc.R,domain_m);
    }

private:
    ippl::NDRegion<T,Dim> domain_m;
    
};

// 1. Species given by attribute
template <class PLayout, typename T, unsigned Dim = 3>
class SecondaryParticleContainer : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    typename Base::particle_position_type P; // Momenta
    ParticleAttrib<unsigned int> Sp; // Particle Species 

    SecondaryParticleContainer(PLayout& pl, unsigned int nsp)
        : Base(pl), nSp(nsp), rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()))
    {
        this->addAttribute(P);
        this->addAttribute(Sp);
    }

    ~SecondaryParticleContainer() {}

    void initialize(Vector_t<T,Dim> rmin_, Vector_t<T,Dim> rmax_, unsigned int work){
        rmin=rmin_;
        rmax=rmax_;
        auto rand_pool64_local_copy = this->rand_pool64;
        for(unsigned int w=0;w<work;++w){
            Kokkos::parallel_for(
                this->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        this->R.getView(), rand_pool64_local_copy, rmin, rmax));
            Kokkos::parallel_for(
                this->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        this->P.getView(), rand_pool64_local_copy, rmin, rmax));
        }
        Kokkos::fence();
    }
    
    void sampleSp(){
        Kokkos::fill_random(this->Sp.getView(), rand_pool64, nSp);
    }

public: // physics
    void spUpdate(double dt, unsigned int sp, unsigned int work){
        auto Rview = this->R.getView();
        auto Pview = this->P.getView();
        auto Spview = this->Sp.getView();
        auto rand_pool64_local_copy = this->rand_pool64;
        Kokkos::parallel_for(this->getLocalNum(), 
        KOKKOS_LAMBDA(const int& i){
            if(Spview(i)==sp){
                auto rand_gen = rand_pool64_local_copy.get_state();
                for(unsigned int w=0;w<work;++w){
                    Rview(i) = Rview(i) + dt * Pview(i);
                    Pview(i) = rand_gen.drand(0.0,1.0); 
                }
                rand_pool64_local_copy.free_state(rand_gen); 
            }
        });    
        //Kokkos::fence();
    }

    void death(double freq, unsigned int sp=0){
        auto Spview = this->Sp.getView();
        auto rand_pool64_local_copy = this->rand_pool64; 
        bool_type lostParticles("lost particles", this->getLocalNum());
        size_type nLost = 0;

        Kokkos::parallel_reduce(this->getLocalNum(),
        KOKKOS_LAMBDA(const int& i, size_type&sum){
            if(Spview(i)==sp){
                auto rand_gen = rand_pool64_local_copy.get_state(); 
                if(rand_gen.drand(0.0,1.0)<freq){
                    lostParticles(i)=1;
                    sum += 1;
                }
                rand_pool64_local_copy.free_state(rand_gen);
            }
        },nLost);
        Kokkos::fence();
        this->destroy(lostParticles, nLost);
    }

    void birth(double freq, unsigned int sp=0){
        auto rand_pool64_local_copy = this->rand_pool64;
        auto Spview = this->Sp.getView();
        size_type new_particles;
       
        Kokkos::parallel_reduce(this->getLocalNum(),
        KOKKOS_LAMBDA(const int& i, size_type& sum){
            if(Spview(i)==sp){
                auto rand_gen = rand_pool64_local_copy.get_state();
                if(rand_gen.drand(0.0,1.0)<freq){
                    sum += 1;
                }
                rand_pool64_local_copy.free_state(rand_gen);
            }
            
        }, new_particles);
        Kokkos::fence();
        this->create(new_particles); 

        Kokkos::parallel_for(Kokkos::RangePolicy(this->getLocalNum()-new_particles-1, this->getLocalNum()),
        KOKKOS_LAMBDA(const int i){
            Spview(i)=sp;
        });
        Kokkos::fence();
    }

private:
    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax;
    unsigned int nSp;
    Kokkos::Random_XorShift64_Pool<> rand_pool64;
};

// 2. Species given by sorted containter
template <class PLayout, typename T, unsigned Dim = 3>
class SortedParticleContainer : public ippl::ParticleBase<PLayout> {
    
    using Base = ippl::ParticleBase<PLayout>;
    using view_type       = Kokkos::View<bool*>;
    using memory_space    = typename view_type::memory_space;
    using execution_space = typename view_type::execution_space;
    using policy_type     = Kokkos::RangePolicy<execution_space>;

public:
    typename Base::particle_position_type P; // Momenta
    
    SortedParticleContainer(PLayout& pl, unsigned int sp)
        : Base(pl), Sp(sp), start(sp+1), end(sp), rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()))
    {
        this->addAttribute(P);
        for(unsigned i=0;i<Sp;++i){
            start[i]=0; // index of the first
            end[i]=0;   // index of the last + 1
        }
    }

    ~SortedParticleContainer() {}
    
    void initialize(Vector_t<T,Dim> rmin_, Vector_t<T,Dim> rmax_, unsigned int work){
        auto rand_pool64_local_copy = this->rand_pool64;
        rmin=rmin_;
        rmax=rmax_;
        for(unsigned i=0;i<Sp;++i){
            for(unsigned int w=0;w<work;++w){
                Kokkos::parallel_for(policy_type(start[i], end[i]), 
                    generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        this->R.getView(), rand_pool64_local_copy, rmin, rmax));
            
                Kokkos::parallel_for(policy_type(start[i], end[i]), 
                    generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        this->P.getView(), rand_pool64_local_copy, rmin, rmax));
            }
        }
        Kokkos::fence();
    }

    void sampleSp(){
        return;
    }

    void create(size_type particles,unsigned int species=0){
        /* 
        Empty particle container -> create evenly spaced, 
        evenly sized particle, buffer regions.

        p = particles
        0 = buffer 
        _____________________________
        |ppp|00|ppp|00|ppp|00|ppp|00|
        –––––––––––––––––––––––––––––

        residual particles (due to integer division) are added to the last species
        */
        if(this->getLocalNum() == 0){
            size_t buffer_size  = particles * (ippl::Comm->getDefaultOverallocation()-1.0) / Sp;
            size_t buffer_res   = particles * static_cast<size_t>(ippl::Comm->getDefaultOverallocation()-1.0) % Sp; 
            size_t part_per_sp  = particles / Sp;
            size_t res          = particles % Sp;
            size_t ptr          = 0;
            for(unsigned int i=0;i<Sp;++i){
               start[i] = ptr;
               end[i] = start[i] + part_per_sp;
               ptr = end[i] + buffer_size;
            }
            end[Sp-1] += res;
            // this is the index after the end of the container == size
            start[Sp] = end[Sp-1] + buffer_size + buffer_res;
            Base::create(particles);
        }
        
        /*
        Filled particle container + small allocation ->
        
        e.g.
        species     = 2
        particles   = 1 
        _____________________________
        |ppp|00|ppp|00|ppp|00|ppp|00|
        –––––––––––––––––––––––––––––
        ==>
        _____________________________
        |ppp|00|ppp|00|pppp|0|ppp|00|
        –––––––––––––––––––––––––––––
        */ 
        
        else if(start[species+1]-end[species]>= particles){
            end[species] += particles; 
            Base::create(particles);
        }

        else{
            size_t buffer_size = (this->getLocalNum() + particles) * (ippl::Comm->getDefaultOverallocation()-1.0) / Sp;
            end[species] += particles;
            size_t newstart = end[species] + buffer_size;
            size_t diff = newstart - start[species+1];
            start[species+1] = newstart;
            create(diff,species+1);
        }

    }

    void getIndexInfo(){
        std::cout << "Indices along the container:\n";
        for(unsigned int i=0;i<Sp; ++i){
            std::cout << "start[" << i << "] = " << start[i] << "  end[" << i << "] = " << end[i] << "\n";
        }
        std::cout << "start[" << Sp << "] = " << start[Sp] << "\n";
    }

    void generateDiagram(const std::string& python_script_path = "generate_diagram.py") const {
        const char* data_filename = "container_data.txt";
        
        // Step 1: Write index data to a file
        std::ofstream data_file(data_filename);
        if (!data_file.is_open()) {
            std::cerr << "Error: Could not open file to write diagram data." << std::endl;
            return;
        }

        data_file << "Indices along the container:\n";
        for (unsigned int i = 0; i < Sp; ++i) {
            data_file << "start[" << i << "] = " << start[i] << "  end[" << i << "] = " << end[i] << "\n";
        }
        data_file << "start[" << Sp << "] = " << start[Sp] << "\n";
        data_file.close();

        // Step 2: Construct and execute the command to call the Python script
        // Note: 'python3' might be needed instead of 'python' on some systems
        std::string command = "python ";
        command += python_script_path;
        command += " ";
        command += data_filename;
        
        std::cout << "Executing command: " << command << std::endl;
        int result = system(command.c_str());

        if (result != 0) {
            std::cerr << "Warning: The python script might have failed. Exit code: " << result << std::endl;
        }
    }

    void destroy(const Kokkos::View<bool*>& invalid,
                 const size_type destroyNum,
                 const unsigned int species) {
        
        PAssert(destroyNum <= end[species]-start[species]);
        
        if(destroyNum == 0){
            return;
        }

        if(destroyNum == end[species]-start[species]){
            end[species] = start[species];
        }

        auto& locDeleteIndex  = Base::deleteIndex_m.template get<memory_space>();
        auto& locKeepIndex    = Base::keepIndex_m.template get<memory_space>();

        // Resize buffers, if necessary
        ippl::detail::runForAllSpaces([&]<typename MemorySpace>() {
            if (Base::attributes_m.template get<memory_space>().size() > 0) {
                int overalloc = ippl::Comm->getDefaultOverallocation();
                auto& del     = Base::deleteIndex_m.template get<memory_space>();
                auto& keep    = Base::keepIndex_m.template get<memory_space>();
                if (del.size() < destroyNum) {
                    Kokkos::realloc(del, destroyNum * overalloc);
                    Kokkos::realloc(keep, destroyNum * overalloc);
                }
            }
        });
        
        // Reset index buffer
        Kokkos::deep_copy(locDeleteIndex, -1);

        // Find the indices of the invalid particles in the valid region
        // In this case the invalid view needs to be full size of the conatiner
        Kokkos::parallel_scan(
            "Scan in ParticleBase::destroy()", policy_type(start[species], end[species] - destroyNum),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && invalid(i)) {
                    locDeleteIndex(idx) = i;
                }
                if (invalid(i)) {
                    idx += 1;
                }
            });
        Kokkos::fence();

        // Determine the total number of invalid particles in the valid region
        size_type maxDeleteIndex = 0;
        Kokkos::parallel_reduce(
            "Reduce in ParticleBase::destroy()", policy_type(0, destroyNum),
            KOKKOS_LAMBDA(const size_t i, size_t& maxIdx) {
                if (locDeleteIndex(i) >= 0 && i > maxIdx) {
                    maxIdx = i;
                }
            },
            Kokkos::Max<size_type>(maxDeleteIndex));

        // Find the indices of the valid particles in the invalid region
        Kokkos::parallel_scan(
            "Second scan in ParticleBase::destroy()",
            Kokkos::RangePolicy<size_type, execution_space>(end[species] - destroyNum, end[species]),
            KOKKOS_LAMBDA(const size_t i, int& idx, const bool final) {
                if (final && !invalid(i)) {
                    locKeepIndex(idx) = i;
                }
                if (!invalid(i)) {
                    idx += 1;
                }
            });

        Kokkos::fence();

        Base::localNum_m -= destroyNum;
        end[species] -= destroyNum;
        
        auto filter = [&]<typename MemorySpace>() {
            return Base::attributes_m.template get<MemorySpace>().size() > 0;
        };
        Base::deleteIndex_m.template copyToOtherSpaces<memory_space>(filter);
        Base::keepIndex_m.template copyToOtherSpaces<memory_space>(filter);

        // Partition the attributes into valid and invalid regions
        // NOTE: The vector elements are pointers, but we want to extract
        // the memory space from the class type, so we explicitly
        // make the lambda argument a pointer to the template parameter
        Base::forAllAttributes([&]<typename Attribute>(Attribute*& attribute) {
            using att_memory_space = typename Attribute::memory_space;
            auto& del              = Base::deleteIndex_m.template get<att_memory_space>();
            auto& keep             = Base::keepIndex_m.template get<att_memory_space>();
            attribute->destroy(del, keep, maxDeleteIndex + 1);
        });
    }

public: // physics

    void spUpdate(double dt, unsigned int sp, unsigned int work){
        auto rand_pool64_local_copy = this->rand_pool64;
        auto Rview = this->R.getView();
        auto Pview = this->P.getView();
        Kokkos::parallel_for(policy_type(start[sp], end[sp]), 
        KOKKOS_LAMBDA(const int i){
            auto rand_gen = rand_pool64_local_copy.get_state();
            for(unsigned int w=0; w<work; ++w){
                Rview(i) = Rview(i) + dt * Pview(i);
                Pview(i) = rand_gen.drand(0.0,1.0);
            }
            rand_pool64_local_copy.free_state(rand_gen); 
        });
        //Kokkos::fence();
    }

    void death(double freq, unsigned int sp=0){
        auto rand_pool64_local_copy = this->rand_pool64;
        bool_type lostParticles("lost particles", start[Sp]);
        size_type nLost = 0;

        Kokkos::parallel_reduce(policy_type(start[sp], end[sp]),
        KOKKOS_LAMBDA(const int& i, size_type&sum){
            auto rand_gen = rand_pool64_local_copy.get_state(); 
            if(rand_gen.drand(0.0,1.0)<freq){
                lostParticles(i)=1;
                sum += 1;
            }
            rand_pool64_local_copy.free_state(rand_gen);
        },nLost);
        Kokkos::fence();
        this->destroy(lostParticles, nLost, sp);
    }
    
    void birth(double freq, unsigned int sp=0){

        auto rand_pool64_local_copy = this->rand_pool64;
        size_type new_particles;
        Kokkos::parallel_reduce(policy_type(start[sp], end[sp]),
        KOKKOS_LAMBDA(const int& , size_type& sum){
            auto rand_gen = rand_pool64_local_copy.get_state();
            if(rand_gen.drand(0.0,1.0)<freq){
                sum += 1;
            }
            rand_pool64_local_copy.free_state(rand_gen);
        }, new_particles);
        Kokkos::fence();
        this->create(new_particles,sp); 
    }

private:
    const unsigned int Sp;
    std::vector<size_t>start;
    std::vector<size_t>end;

    Vector_t<double, Dim> rmin;
    Vector_t<double, Dim> rmax; 
    Kokkos::Random_XorShift64_Pool<> rand_pool64;

    //Vector_t<bool, 1> lost_particles; 
};

// 3. Supecontainer with seperate containers for each species
template <class PLayout, typename T, unsigned Dim = 3>
class SuperContainer{
    
    using Base = ippl::ParticleBase<PLayout>;
    
    class ParticleContainer : public ippl::ParticleBase<PLayout> {
    
    public:
        typename Base::particle_position_type P; 
        
        ParticleContainer(PLayout& pl) : Base(pl)
        {
            this->addAttribute(P);
        }

        ~ParticleContainer() {}
    private:

    };

public:
    SuperContainer(PLayout& pl, unsigned int nsp)
        : nSp(nsp), rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()))
    {
        for(unsigned i=0;i<nSp;++i){
            species_m.push_back(std::make_unique<ParticleContainer>(pl));
        }
    }

    void sampleSp(){
        return;
    }
public: // physics
    void create(size_type particles){
        for(unsigned i=0;i<nSp;++i){
            species_m[i]->create(particles/nSp);
        }
    }

    void create(size_type particles, unsigned int species){
        species_m[species]->create(particles);
    }

    void initialize(Vector_t<T,Dim> rmin_, Vector_t<T,Dim> rmax_, unsigned int work){
        auto rand_pool64_local_copy = this->rand_pool64;
        for(unsigned i=0;i<nSp;++i){
            for(unsigned int w=0;w<work;++w){
                Kokkos::parallel_for(species_m[i]->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        species_m[i]->R.getView(), rand_pool64_local_copy, rmin_, rmax_));
                Kokkos::parallel_for(species_m[i]->getLocalNum(), generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                        species_m[i]->P.getView(), rand_pool64_local_copy, rmin_, rmax_));
            }
       }
        Kokkos::fence();
    }

    void spUpdate(double dt, unsigned int sp, unsigned int work){
        auto rand_pool64_local_copy = this->rand_pool64;
        auto Rview = species_m[sp]->R.getView();
        auto Pview = species_m[sp]->P.getView();
        Kokkos::parallel_for(species_m[sp]->getLocalNum(), 
        KOKKOS_LAMBDA(const int& i){
            auto rand_gen = rand_pool64_local_copy.get_state();
            for(unsigned int w=0; w<work; ++w){
                Rview(i) = Rview(i) + dt * Pview(i);
                Pview(i) = rand_gen.drand(0.0,1.0);
            }
            rand_pool64_local_copy.free_state(rand_gen);
        });    
        //Kokkos::fence();
    }

    void death(double freq, unsigned int sp){
        auto rand_pool64_local_copy = this->rand_pool64;
        bool_type lostParticles("lost particles", species_m[sp]->getLocalNum());
        size_type nLost = 0;

        Kokkos::parallel_reduce(species_m[sp]->getLocalNum(),
        KOKKOS_LAMBDA(const int& i, size_type&sum){
            auto rand_gen = rand_pool64_local_copy.get_state(); 
            if(rand_gen.drand(0.0,1.0)<freq){
                lostParticles(i)=1;
                sum += 1;
            }
            rand_pool64_local_copy.free_state(rand_gen);
            
        },nLost);
        Kokkos::fence();
        species_m[sp]->destroy(lostParticles, nLost);
    }

    void birth(double freq, unsigned int sp){
        auto rand_pool64_local_copy = this->rand_pool64;
        size_type new_particles;
        Kokkos::parallel_reduce(species_m[sp]->getLocalNum(),
        KOKKOS_LAMBDA(const int&, size_type& sum){
            auto rand_gen = rand_pool64_local_copy.get_state();
            if(rand_gen.drand(0.0,1.0)<freq){
                sum += 1;
            }
            rand_pool64_local_copy.free_state(rand_gen);
            
        }, new_particles);
        Kokkos::fence();
        species_m[sp]->create(new_particles); 
    }

private:
    unsigned int nSp;
    std::vector<std::unique_ptr<ParticleContainer>> species_m;
    Kokkos::Random_XorShift64_Pool<> rand_pool64;
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Message objects
        Inform msg("ParticleContainerTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();

        // Parse arguments: 
        int arg=1;
        unsigned int nSp        = std::atoi(argv[arg++]); 
        size_type ppSp          = std::atoll(argv[arg++]);
        double b_freq           = std::atof(argv[arg++]);
        double d_freq           = std::atof(argv[arg++]);
        const double dt         = 1.0;

        // Define Domain
        Vector_t<double, Dim> rmin(-1.0);
        Vector_t<double, Dim> rmax(1.0); 
        ippl::NDRegion<double,3> domain;
        for(unsigned int d=0;d<Dim;++d){
            domain[d] = ippl::PRegion<double>(-1,1);
        }
        
        // Particle Layout
        SecondaryParticleLayout<double,Dim> PL;
        PL.setDomain(domain);
        PL.setParticleBC(ippl::BC::PERIODIC);
        
        // Timers
        static IpplTimings::TimerRef mainTimer          = IpplTimings::getTimer("total");
        static IpplTimings::TimerRef allUpdate          = IpplTimings::getTimer("allUpdate");
        static IpplTimings::TimerRef specific           = IpplTimings::getTimer("specific");
        static IpplTimings::TimerRef birth              = IpplTimings::getTimer("birth");
        static IpplTimings::TimerRef death              = IpplTimings::getTimer("death");
       
        // Particle Container Pointer
        //using container_type = SecondaryParticleContainer<SecondaryParticleLayout<double,Dim>, double, Dim>; 
        //using container_type = SortedParticleContainer<SecondaryParticleLayout<double,Dim>, double, Dim>; 
        using container_type = SuperContainer<SecondaryParticleLayout<double,Dim>, double, Dim>; 
       
        msg << "Starting iterations..." << endl;
        for(unsigned rep=0;rep<10;++rep){
            std::shared_ptr<container_type> PC = std::make_shared<container_type>(PL, nSp);
            
            // Initialize Particles
            PC->create(nSp * ppSp);
            PC->initialize(rmin,rmax,1);
            PC->sampleSp();
            
            // Main iteration loop
            IpplTimings::startTimer(mainTimer);
            for(unsigned int it=0; it<20; it++){ 
                std::cout << "Iteration " << it << std::endl;
                
                // All particle update
                IpplTimings::startTimer(allUpdate);
                PC->initialize(rmin,rmax,20);
                IpplTimings::stopTimer(allUpdate);

                // Species-specific deterministic
                IpplTimings::startTimer(specific);
                for(unsigned i=0;i<nSp;++i){
                    PC->spUpdate(dt, i,20);
                }
                Kokkos::fence(); 
                IpplTimings::stopTimer(specific);
                
                // birth/death
                for(unsigned i=0;i<nSp;++i){
                    IpplTimings::startTimer(death);
                    PC->death(d_freq, i);
                    IpplTimings::stopTimer(death);
                    IpplTimings::startTimer(birth);
                    PC->birth(b_freq, i);
                    IpplTimings::stopTimer(birth);
                }
            }
            IpplTimings::stopTimer(mainTimer);
        }
        
        

        msg << "Particle Container Test: End. " << endl;
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> time_chrono =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        std::cout << "Elapsed time: " << time_chrono.count() << std::endl;

        //PC->generateDiagram();
    }
    ippl::finalize();
    return 0;
}



