// IndependentParticlesTest
// //   Usage:
//     srun ./IndependentParticlesTest

#include <Kokkos_Random.hpp>
#include <chrono>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>
#include <cassert>
#include <array>
#include <fstream>    
#include <set> 

#include "Utility/IpplTimings.h"

#include "ChargedParticles.hpp"
#include "Utility/TypeUtils.h"

constexpr unsigned Dim = 3;

const char* TestName = "IndependentParticleTest";

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

template <typename size_type>
class Node {
public:
    Node() = default; 
    Node(size_type depth, std::shared_ptr<Node<size_type>> parent, size_type numleaves): 
        depth_m(depth), 
        numLeaves_m(numleaves){
            if(depth!=0){
                //parent_m=std::make_shared<Node<size_type>>(*parent);
                parent_m=parent;
            }
            else{
                parent_m=nullptr;
            }
            numChildren_m=0;
            for (size_type i = 0; i < 3; i++){
                children_m[i]=nullptr;
            }
            
        } 
private:
    size_type order_m;
    size_type depth_m;
    size_type numLeaves_m;
    std::shared_ptr<Node<size_type>> parent_m;
    std::array<std::shared_ptr<Node<size_type>>, 3> children_m;
    size_type numChildren_m;
public:
    size_type getDepth(){
        return this->depth_m;
    }
    size_type getNumLeaves(){
        return this->numLeaves_m;
    }
    void addChild(std::shared_ptr<Node<size_type>> child_ptr){
        this->children_m[numChildren_m]=child_ptr;
        this->numChildren_m++;
    }
    void setOrder(size_type order){
        this->order_m=order;
    }
    size_type getNumChildren(){
        return this->numChildren_m;
    }
    std::array<std::shared_ptr<Node<size_type>>,3> getChildren(){
        return this->children_m;
    }
    size_type getOrder() const {
        return this->order_m;
    }
    std::shared_ptr<Node<size_type>> getParent() const { 
        return this->parent_m;
    }
};

//template <typename size_type=size_type>
class Tree{
public: // constructors
    Tree() = default;
    Tree(size_type numNodes): numNodes_m(numNodes){
        determineBinDepth();
        determineNumLeaves();    
        //std::cout << "Calculated binDepth = " << binDepth_m << std::endl;
        //std::cout << "Calculated numLeafNodes_to_distribute for root = " << numLeafNodes_m << std::endl;
        root_m = std::make_shared<Node<size_type>>(0, nullptr, numLeafNodes_m);
        createChildren(root_m);
        determineOrder(root_m); 
    }

public: // member functions
    void determineBinDepth(){
        size_type depth=0;
        size_type count=0;
        while(count<this->numNodes_m){
            depth++;
            count=(2<<(depth-1))-1+(3<<(depth-1));
        }
        this->binDepth_m = depth-1;
    }
    void determineNumLeaves(){
        this->numLeafNodes_m=this->numNodes_m-(2<<this->binDepth_m)+1;
    }
    void createChildren(std::shared_ptr<Node<size_type>> parent){
        if(parent->getDepth()<this->binDepth_m){
            std::shared_ptr<Node<size_type>> rChild_ptr;
            std::shared_ptr<Node<size_type>> lChild_ptr;
            size_type temp=3<<(this->binDepth_m-parent->getDepth()-1);
            if(temp >= parent->getNumLeaves()){
                rChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth()+1, parent, parent->getNumLeaves());
                lChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth()+1, parent, 0);
            }
            else if(temp < parent->getNumLeaves()){
                rChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth()+1, parent, temp);
                lChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth()+1, parent, parent->getNumLeaves()-temp);
            }
            else{
                assert(false && "error");
            }
            parent->addChild(lChild_ptr);
            parent->addChild(rChild_ptr);
            createChildren(lChild_ptr);    
            createChildren(rChild_ptr);    
            return;
        }
        else if(parent->getDepth()==this->binDepth_m){
            for (size_type i = 0; i < parent->getNumLeaves(); i++){
                std::shared_ptr<Node<size_type>> leafNode_ptr=std::make_shared<Node<size_type>>(parent->getDepth()+1, parent, 0);
                parent->addChild(leafNode_ptr);
            }
            return;
        }
        else{
            assert(false && "ERROR");
        }
    }
    
    void determineOrder(std::shared_ptr<Node<size_type>> node){
        static size_type current=0;
        node->setOrder(current);
        bool isCurrent=false;
        if(ippl::Comm->rank()==current){
            if(current != 0){
                globalParent_m=node->getParent()->getOrder();
            }
            isCurrent=true;
            globalNumChildren_m=node->getNumChildren();
        }
        current++;
        for(size_type i=0; i<node->getNumChildren();++i){
            if(isCurrent){
                globalChildren_m[i]=current;
            }
            determineOrder(node->getChildren()[i]);
        }
        return;
    }

public: // getters & setters
std::shared_ptr<Node<size_type>> getRoot(){
    return root_m;
}
size_type getGlobalParent(){
    return globalParent_m;
}
std::array<size_type,3> getGlobalChildren(){
    return globalChildren_m;
}
size_type getGlobalNumChildren(){
    return globalNumChildren_m;
}
public: // visualization

// --- Graphviz Visualization Functions ---
// Color based on groups of 4 by order
void generateGraphvizRecursive(std::shared_ptr<Node<size_type>>node, std::ofstream& dotFile, std::set<std::shared_ptr<Node<size_type>>>& visitedNodes, const std::vector<std::string>& colorPalette) {
    if (!node || visitedNodes.count(node)) {
        return;
    }
    visitedNodes.insert(node);

    // Determine fill color based on node order (groups of 4)
    std::string fillColor = "lightgray"; // Default if palette is empty
    if (!colorPalette.empty()) {
        // Group nodes by their order / 4
        // Nodes with order 0,1,2,3 get group 0
        // Nodes with order 4,5,6,7 get group 1
        // ...and so on.
        size_t colorGroupIndex = node->getOrder() / 4;
        fillColor = colorPalette[colorGroupIndex % colorPalette.size()];
    }

    // Define the node in DOT format
    dotFile << "  node" << node->getOrder() // Use order for unique node ID in DOT
            << " [label=\"Order: " << node->getOrder()
            << "\\nDepth: " << node->getDepth()
            << "\\nLeavesToDist: " << node->getNumLeaves();
    if (node->getParent()) {
        // Check if parent has an order assigned (it should if orderTree was called properly)
        dotFile << "\\nParentOrder: " << node->getParent()->getOrder();
    }
    // Apply the chosen fill color
    dotFile << "\", fillcolor=\"" << fillColor << "\"];\n";


    // Define edges to children and recurse
    std::array<std::shared_ptr<Node<size_type>>, 3> children = node->getChildren(); // Gets a copy
    for (size_type i = 0; i < node->getNumChildren(); ++i) {
        std::shared_ptr<Node<size_type>>child = children[i];
        if (child) {
            // Ensure child also has an order for its ID
            dotFile << "  node" << node->getOrder() << " -> node" << child->getOrder() << ";\n";
            // Pass the colorPalette down
            generateGraphvizRecursive(child, dotFile, visitedNodes, colorPalette);
        }
    }
}

void generateGraphvizOutput(std::shared_ptr<Node<size_type>>rootNode, const std::string& filename) {
    std::ofstream dotFile(filename);
    if (!dotFile.is_open()) {
        std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
        return;
    }

    const std::vector<std::string> colorPalette = {
        "skyblue", "palegreen", "khaki", "lightcoral", "mediumpurple",
        "gold", "aquamarine", "salmon", "orchid", "sandybrown"
    };

    dotFile << "digraph Tree {\n";
    dotFile << "  graph [rankdir=TB];\n";
    dotFile << "  node [shape=box, style=filled, fillcolor=lightgray]; // Default node style\n";
    dotFile << "  edge [arrowhead=vee];\n";

    std::set<std::shared_ptr<Node<size_type>>> visitedNodes;
    if (rootNode) {
        generateGraphvizRecursive(rootNode, dotFile, visitedNodes, colorPalette);
    }

    dotFile << "}\n";
    dotFile.close();

    std::cout << "\nGraphviz DOT file generated: " << filename << std::endl;
    std::cout << "To generate a PNG image, run: dot -Tpng " << filename << " -o tree.png" << std::endl;
}
// --- End Graphviz Visualization Functions ---

private: // tree variables
    std::shared_ptr<Node<size_type>>root_m;
    size_type numNodes_m;
    size_type binDepth_m;
    size_type numLeafNodes_m;
    size_type globalParent_m;
    std::array<size_type, 3> globalChildren_m;
    size_type globalNumChildren_m;
};

template <typename T, unsigned Dim, typename... PositionProperties>
class ParticleTreeLayout : public ippl::detail::ParticleLayout<T, Dim, PositionProperties...> {

public: // types
    using Base = ippl::detail::ParticleLayout<T, Dim, PositionProperties...>;
    using typename Base::position_memory_space, typename Base::position_execution_space;
    using hash_type   = ippl::detail::hash_type<position_memory_space>;
    using bool_type   = typename ippl::detail::ViewType<bool, 1, position_memory_space>::view_type;
    
public: // constructor
    ParticleTreeLayout()
            : ippl::detail::ParticleLayout<T, Dim, PositionProperties...>() 
            {
                Tree commTree(ippl::Comm->size());
                //commTree.generateGraphvizOutput(commTree.getRoot(), "tree.dot");
                parentRank_m=commTree.getGlobalParent();
                childrenRanks_m=commTree.getGlobalChildren();
                numChildren_m=commTree.getGlobalNumChildren();
                /*
                std::cout<<"Rank "<<ippl::Comm->rank()<<" has parent "<<parentRank_m
                <<" and children ";
                for(size_type i=0; i<numChildren_m;++i){
                    std::cout<<childrenRanks_m[i]<<" ";
                }
                std::cout<<std::endl;
                */
            }

public: // member functions
   
    template <class ParticleContainer>
    void loadbalance(ParticleContainer& pc){
        
        // global work reduction
        int subtreeWork = 0;
        std::array<int, 3> childrenSubtreeWork;
        reduceWorkLoad(pc.getLocalNum(), subtreeWork, childrenSubtreeWork);
       
        // w_avg = data[0]
        // R     = data[1]
        std::array<int,2> data;
        
        if(ippl::Comm->rank()==0){
            data[0]=subtreeWork/ippl::Comm->size();
            data[1]=subtreeWork%ippl::Comm->size();
        }
        MPI_Bcast(data.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
        
        // calculate quotas and subtree quotas
        int quota; // q_i
        int subtreeQuota; // Q_i
        std::array<int, 3> childrenSubtreeQuotas; // Q_j where j is child of i
        computeQuota(quota, subtreeQuota, childrenSubtreeQuotas, data); 
        
        // distribute work
        exchangeWork(subtreeWork, subtreeQuota, childrenSubtreeQuotas, childrenSubtreeWork, pc);
        std::cout<<"Rank "<<ippl::Comm->rank()<<" has finished" <<std::endl;
    }

    void reduceWorkLoad(int localparticles, int& subtreework, auto& childrensubtreework){
        subtreework=localparticles;
        for(size_type i=0;i<numChildren_m;++i){
            int temp;
            MPI_Status status;
            MPI_Recv(&temp, 1, MPI_INT,childrenRanks_m[i], childrenRanks_m[i], MPI_COMM_WORLD, &status);
            subtreework += temp;
            childrensubtreework[i]=temp;
        }
        if(ippl::Comm->rank() != 0){
            MPI_Request request;
            MPI_Isend(&subtreework, 1, MPI_INT,parentRank_m,ippl::Comm->rank(),MPI_COMM_WORLD, &request);
        }
    }
    void computeQuota(int& quota, int& subtreeQuota, auto& childrenSubtreeQuotas, auto& data){
        if(ippl::Comm->rank()<data[1]){
            quota=data[0]+1;
        }
        else{
            quota=data[0];
        }       
        subtreeQuota=quota;

        if(numChildren_m>0){
            for(size_type i=0;i<numChildren_m;++i){
                MPI_Status status;
                MPI_Recv(&childrenSubtreeQuotas[i], 1, MPI_INT, childrenRanks_m[i], childrenRanks_m[i], MPI_COMM_WORLD, &status);
                subtreeQuota += childrenSubtreeQuotas[i];
            }
        }
        if(ippl::Comm->rank()!=0){
            MPI_Request request;
            MPI_Isend(&subtreeQuota, 1, MPI_INT, parentRank_m, ippl::Comm->rank(),MPI_COMM_WORLD, &request);
        }
        return;
    }

    template <class ParticleContainer>
    void exchangeWork(int& subtreeWork, int& subtreeQuota, auto& childrenSubtreeQuotas, auto& childrenSubtreeWork, ParticleContainer& pc){
        
        //int tag = ippl::Comm->next_tag(mpi::tag::P_SPATIAL_LAYOUT, mpi::tag::P_LAYOUT_CYCLE); 
        int tag = 0;
       
        if(subtreeWork < subtreeQuota && ippl::Comm->rank() != 0){
            // receive particles from partent
            size_t nRecvs = subtreeQuota-subtreeWork;
            std::cout<<"Rank "<<ippl::Comm->rank()<<" starting receive from partent "<< parentRank_m <<std::endl;
            pc.recvFromRank(parentRank_m, tag, nRecvs);
            std::cout<<"Rank "<<ippl::Comm->rank()<<" has received from partent "<< parentRank_m <<std::endl;
        }
        
        for(size_type i=0; i<numChildren_m; ++i){
            if(childrenSubtreeWork[i] > childrenSubtreeQuotas[i]){
                // receive particles from children
                std::cout<<"Rank "<<ippl::Comm->rank()<<" starting receive from child "<< childrenRanks_m[i] <<std::endl;
                size_t nRecvs = childrenSubtreeWork[i] - childrenSubtreeQuotas[i];
                pc.recvFromRank(childrenRanks_m[i], tag, nRecvs);
                std::cout<<"Rank "<<ippl::Comm->rank()<<" has received from child "<< childrenRanks_m[i] <<std::endl;
            }
        }
        
        if(subtreeWork > subtreeQuota){
            // Number of particles to send away
            size_t numInvParticles=subtreeWork-subtreeQuota;
            
            // Send particles to parent rank
            std::vector<MPI_Request> requests(0);
            hash_type hash("hash", numInvParticles);
            fillHash(numInvParticles, hash, pc);
            pc.sendToRank(parentRank_m, tag, requests, hash);
            std::cout<<"Rank "<<ippl::Comm->rank()<<" sent to partent "<< parentRank_m <<std::endl;
            // Delete invalid particles
            bool_type invalidParticles("validity of particles", pc.getLocalNum());
            fillInvalid(numInvParticles, invalidParticles, pc);
            pc.internalDestroy(invalidParticles, numInvParticles);
        }
        for(size_type i=0; i<numChildren_m; ++i){
            if(childrenSubtreeWork[i] < childrenSubtreeQuotas[i]){
                // Number of particles to send away
                size_t numInvParticles=childrenSubtreeQuotas[i]-childrenSubtreeWork[i];

                // Send particles to parent child rank
                std::vector<MPI_Request> requests(0);
                hash_type hash("hash", numInvParticles);
                fillHash(numInvParticles, hash, pc);
                pc.sendToRank(childrenRanks_m[i], tag, requests, hash);
                std::cout<<"Rank "<<ippl::Comm->rank()<<" sent to child "<< childrenRanks_m[i] <<std::endl; 
                // Delete invalid particles
                bool_type invalidParticles("validity of particles", pc.getLocalNum());
                fillInvalid(numInvParticles, invalidParticles, pc);
                pc.internalDestroy(invalidParticles, numInvParticles);
            }
        }
        std::cout<<"Rank "<<ippl::Comm->rank()<<" has sent "<< std::endl;

    }

    /* Fill the hash with the last nParticles particles in the container */
    template <class ParticleContainer>
    void fillHash(int nParticles, hash_type& hash, ParticleContainer& pc) {
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_for(
            "ParticleTreeLayout::fillHash()", policy_type(pc.getLocalNum()-nParticles, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i){
                hash(i-pc.getLocalNum()+nParticles)=i;
            }
        );
    }

    /* Determine invalid particles */
    template <class ParticleContainer>
    void fillInvalid(int nParticles, bool_type& invalidParticles, ParticleContainer& pc){
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_for(
            "ParticleTreeLayout::fillInvalid()", policy_type(0, pc.getLocalNum()), 
            KOKKOS_LAMBDA(const size_t i){
                bool isInvalid = i >= pc.getLocalNum()-nParticles;
                invalidParticles(i) = 0 + 1 * isInvalid;
            }
        );
    }
private:
    size_type parentRank_m;
    std::array<size_type,3> childrenRanks_m;
    size_type numChildren_m;
};

template <class PLayout, typename T, unsigned Dim = 3>
class IndependentParticles : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    // Domain
    Vector_t<double, Dim> rmin_m;
    Vector_t<double, Dim> rmax_m;

    // Charge, Mass, Time, Velocity Attributes
    ParticleAttrib<double> q;
    ParticleAttrib<double> m;
    ParticleAttrib<double> t;
    ParticleAttrib<double> v;

    // Constructor
    IndependentParticles(PLayout& pl, Vector_t<double, Dim> rmin, Vector_t<double, Dim> rmax)
        : Base(pl)
        , rmin_m(rmin)
        , rmax_m(rmax)
    {
        // Register Attributes
        this->addAttribute(q);
        this->addAttribute(m);
        this->addAttribute(t);
        this->addAttribute(v);
    }

};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        // Inform objects for output
        Inform msg("IndependentParticleTest");
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        auto start = std::chrono::high_resolution_clock::now();

        // Main Timer
        static IpplTimings::TimerRef mainTimer = IpplTimings::getTimer("total");
        IpplTimings::startTimer(mainTimer);
        msg << "Independent Particles Test" << endl;
        
        
        // Particle Bunch Type
        using bunch_type =
            IndependentParticles<ParticleTreeLayout<double,Dim>,double>;
        std::unique_ptr<bunch_type> P;

        // create mesh and layout objects for this problem domain
        Vector_t<double, Dim> rmin(0.0);
        Vector_t<double, Dim> rmax(20.0);

        const double t_0 = 0.0;

        // Initialize Particle Bunch
        ParticleTreeLayout<double, Dim> PL;
        P = std::make_unique<bunch_type>(PL, rmin, rmax);

        // Create Particles on each Rank
        size_type nloc = 10*ippl::Comm->rank();
        P->create(nloc);
        std::cout<< "Rank "<< ippl::Comm->rank() << " created particles "<< nloc << std::endl;
        P->getLayout().loadbalance(*P);
        
        std::cout<<"Rank "<<ippl::Comm->rank()<<" has "<<P->getLocalNum()<<std::endl;
        // Sample Particle Initial Locations
        /*
        Kokkos::Random_XorShift64_Pool<> rand_pool64((size_type)(42 + 100 * ippl::Comm->rank()));
        Kokkos::parallel_for(
            nloc, generate_random<Vector_t<double, Dim>, Kokkos::Random_XorShift64_Pool<>, Dim>(
                      P->R.getView(), rand_pool64, rmin, rmax));
        Kokkos::fence();

        using RandomPool = Kokkos::Random_XorShift64_Pool<Kokkos::DefaultExecutionSpace>;
        RandomPool rand_pool(12345);

        msg << "particles created and initial conditions assigned " << endl;

        // Begin Particle Tracks
        msg << "Starting Tracks..." << endl;
        Kokkos::parallel_for(P->getLocalNum(), 
            KOKKOS_LAMBDA(const int i){
                // NOTE: Dummy Work
                RandomPool::generator_type steps_generator = rand_pool.get_state();
                int steps = static_cast<int>(steps_generator.drand(0.0, 1.0) * 10000);
                volatile double dummy_result = 0.0; // Volatile variable
                for (int w = 0; w < steps; ++w) {
                    dummy_result += Kokkos::sin(static_cast<double>(w) * 0.01 + i * 0.001);
                }
                rand_pool.free_state(steps_generator);
            }
        );
        */
        // TODO: MPI WaitAll
        // TODO: Load Balancing
        // TODO: Restart Loop with new Particles
        
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
