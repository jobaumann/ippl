#ifndef PARTICLE_TREE_LAYOUT_HPP
#define PARTICLE_TREE_LAYOUT_HPP

#include <array>
#include <cassert>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "Ippl.h"
#include "Utility/TypeUtils.h"

using size_type = ippl::detail::size_type;

template <typename T, unsigned Dim = 3>
using Vector_t = ippl::Vector<T, Dim>;

template <typename T>
using ParticleAttrib = ippl::ParticleAttrib<T>;

// ============================================================================
// Node: Represents a node in the communication tree
// ============================================================================
template <typename size_type>
class Node {
public:
    Node() = default;
    Node(size_type depth, std::shared_ptr<Node<size_type>> parent, size_type numleaves)
        : depth_m(depth)
        , numLeaves_m(numleaves) {
        if (depth != 0) {
            parent_m = parent;
        } else {
            parent_m = nullptr;
        }
        numChildren_m = 0;
        for (size_type i = 0; i < 3; i++) {
            children_m[i] = nullptr;
        }
    }

    size_type getDepth() { return this->depth_m; }
    size_type getNumLeaves() { return this->numLeaves_m; }
    size_type getNumChildren() { return this->numChildren_m; }
    size_type getOrder() const { return this->order_m; }
    std::shared_ptr<Node<size_type>> getParent() const { return this->parent_m; }
    std::array<std::shared_ptr<Node<size_type>>, 3> getChildren() { return this->children_m; }

    void addChild(std::shared_ptr<Node<size_type>> child_ptr) {
        this->children_m[numChildren_m] = child_ptr;
        this->numChildren_m++;
    }

    void setOrder(size_type order) { this->order_m = order; }

private:
    size_type order_m;
    size_type depth_m;
    size_type numLeaves_m;
    std::shared_ptr<Node<size_type>> parent_m;
    std::array<std::shared_ptr<Node<size_type>>, 3> children_m;
    size_type numChildren_m;
};

// ============================================================================
// Tree: Builds an almost-binary tree with ternary bottom layer
// ============================================================================
class Tree {
public:
    Tree() = default;
    Tree(size_type numNodes)
        : numNodes_m(numNodes) {
        determineBinDepth();
        determineNumLeaves();
        root_m = std::make_shared<Node<size_type>>(0, nullptr, numLeafNodes_m);
        createChildren(root_m);
        determineOrder(root_m);
    }

    std::shared_ptr<Node<size_type>> getRoot() { return root_m; }
    size_type getGlobalParent() { return globalParent_m; }
    std::array<size_type, 3> getGlobalChildren() { return globalChildren_m; }
    size_type getGlobalNumChildren() { return globalNumChildren_m; }

    void generateGraphvizOutput(std::shared_ptr<Node<size_type>> rootNode,
                                const std::string& filename) {
        std::ofstream dotFile(filename);
        if (!dotFile.is_open()) {
            std::cerr << "Error: Could not open " << filename << " for writing." << std::endl;
            return;
        }

        const std::vector<std::string> colorPalette = {"skyblue",   "palegreen", "khaki",
                                                       "lightcoral", "mediumpurple", "gold",
                                                       "aquamarine", "salmon",    "orchid",
                                                       "sandybrown"};

        dotFile << "digraph Tree {\n";
        dotFile << "  graph [rankdir=TB];\n";
        dotFile << "  node [shape=box, style=filled, fillcolor=lightgray];\n";
        dotFile << "  edge [arrowhead=vee];\n";

        std::set<std::shared_ptr<Node<size_type>>> visitedNodes;
        if (rootNode) {
            generateGraphvizRecursive(rootNode, dotFile, visitedNodes, colorPalette);
        }

        dotFile << "}\n";
        dotFile.close();

        std::cout << "\nGraphviz DOT file generated: " << filename << std::endl;
        std::cout << "To generate a PNG image, run: dot -Tpng " << filename << " -o tree.png"
                  << std::endl;
    }

private:
    void determineBinDepth() {
        size_type depth = 0;
        size_type count = 0;
        while (count < this->numNodes_m) {
            depth++;
            count = (2 << (depth - 1)) - 1 + (3 << (depth - 1));
        }
        this->binDepth_m = depth - 1;
    }

    void determineNumLeaves() {
        this->numLeafNodes_m = this->numNodes_m - (2 << this->binDepth_m) + 1;
    }

    void createChildren(std::shared_ptr<Node<size_type>> parent) {
        if (parent->getDepth() < this->binDepth_m) {
            std::shared_ptr<Node<size_type>> rChild_ptr;
            std::shared_ptr<Node<size_type>> lChild_ptr;
            size_type temp = 3 << (this->binDepth_m - parent->getDepth() - 1);
            if (temp >= parent->getNumLeaves()) {
                rChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth() + 1, parent,
                                                                parent->getNumLeaves());
                lChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth() + 1, parent, 0);
            } else if (temp < parent->getNumLeaves()) {
                rChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth() + 1, parent, temp);
                lChild_ptr = std::make_shared<Node<size_type>>(parent->getDepth() + 1, parent,
                                                                parent->getNumLeaves() - temp);
            } else {
                assert(false && "error");
            }
            parent->addChild(lChild_ptr);
            parent->addChild(rChild_ptr);
            createChildren(lChild_ptr);
            createChildren(rChild_ptr);
            return;
        } else if (parent->getDepth() == this->binDepth_m) {
            for (size_type i = 0; i < parent->getNumLeaves(); i++) {
                std::shared_ptr<Node<size_type>> leafNode_ptr =
                    std::make_shared<Node<size_type>>(parent->getDepth() + 1, parent, 0);
                parent->addChild(leafNode_ptr);
            }
            return;
        } else {
            assert(false && "ERROR");
        }
    }

    void determineOrder(std::shared_ptr<Node<size_type>> node) {
        static size_type current = 0;
        node->setOrder(current);
        bool isCurrent = false;
        if (ippl::Comm->rank() == current) {
            if (current != 0) {
                globalParent_m = node->getParent()->getOrder();
            }
            isCurrent             = true;
            globalNumChildren_m = node->getNumChildren();
        }
        current++;
        for (size_type i = 0; i < node->getNumChildren(); ++i) {
            if (isCurrent) {
                globalChildren_m[i] = current;
            }
            determineOrder(node->getChildren()[i]);
        }
        return;
    }

    void generateGraphvizRecursive(std::shared_ptr<Node<size_type>> node, std::ofstream& dotFile,
                                   std::set<std::shared_ptr<Node<size_type>>>& visitedNodes,
                                   const std::vector<std::string>& colorPalette) {
        if (!node || visitedNodes.count(node)) {
            return;
        }
        visitedNodes.insert(node);

        std::string fillColor = "lightgray";
        if (!colorPalette.empty()) {
            size_t colorGroupIndex = node->getOrder() / 4;
            fillColor              = colorPalette[colorGroupIndex % colorPalette.size()];
        }

        dotFile << "  node" << node->getOrder() << " [label=\"Order: " << node->getOrder()
                << "\\nDepth: " << node->getDepth()
                << "\\nLeavesToDist: " << node->getNumLeaves();
        if (node->getParent()) {
            dotFile << "\\nParentOrder: " << node->getParent()->getOrder();
        }
        dotFile << "\", fillcolor=\"" << fillColor << "\"];\n";

        std::array<std::shared_ptr<Node<size_type>>, 3> children = node->getChildren();
        for (size_type i = 0; i < node->getNumChildren(); ++i) {
            std::shared_ptr<Node<size_type>> child = children[i];
            if (child) {
                dotFile << "  node" << node->getOrder() << " -> node" << child->getOrder()
                        << ";\n";
                generateGraphvizRecursive(child, dotFile, visitedNodes, colorPalette);
            }
        }
    }

private:
    std::shared_ptr<Node<size_type>> root_m;
    size_type numNodes_m;
    size_type binDepth_m;
    size_type numLeafNodes_m;
    size_type globalParent_m;
    std::array<size_type, 3> globalChildren_m;
    size_type globalNumChildren_m;
};

// ============================================================================
// ParticleTreeLayout: Particle layout with tree-based load balancing
// ============================================================================
template <typename T, unsigned Dim, typename... PositionProperties>
class ParticleTreeLayout : public ippl::detail::ParticleLayout<T, Dim, PositionProperties...> {
public:
    using Base = ippl::detail::ParticleLayout<T, Dim, PositionProperties...>;
    using typename Base::position_memory_space, typename Base::position_execution_space;
    using hash_type = ippl::detail::hash_type<position_memory_space>;
    using bool_type =
        typename ippl::detail::ViewType<bool, 1, position_memory_space>::view_type;

    ParticleTreeLayout(bool debug = false)
        : ippl::detail::ParticleLayout<T, Dim, PositionProperties...>()
        , debug_m(debug) {
        Tree commTree(ippl::Comm->size());

        // Generate tree visualization if debug mode enabled
        if (debug_m && ippl::Comm->rank() == 0) {
            commTree.generateGraphvizOutput(commTree.getRoot(), "tree_structure.dot");
        }

        parentRank_m    = commTree.getGlobalParent();
        childrenRanks_m = commTree.getGlobalChildren();
        numChildren_m   = commTree.getGlobalNumChildren();

        // Print tree structure for each rank if debug enabled
        if (debug_m) {
            std::cout << "Rank " << ippl::Comm->rank() << ": parent=" << parentRank_m
                      << ", numChildren=" << numChildren_m << ", children=[";
            for (size_type i = 0; i < numChildren_m; ++i) {
                std::cout << childrenRanks_m[i];
                if (i < numChildren_m - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    template <class ParticleContainer>
    void loadbalance(ParticleContainer& pc) {
        // Phase 1: Global work reduction
        int subtreeWork = 0;
        std::array<int, 3> childrenSubtreeWork;
        reduceWorkLoad(pc.getLocalNum(), subtreeWork, childrenSubtreeWork);

        // Broadcast average work and remainder
        std::array<int, 2> data;
        if (ippl::Comm->rank() == 0) {
            data[0] = subtreeWork / ippl::Comm->size();  // w_avg
            data[1] = subtreeWork % ippl::Comm->size();  // R
        }
        MPI_Bcast(data.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);

        // Phase 2: Calculate quotas and subtree quotas
        int quota;
        int subtreeQuota;
        std::array<int, 3> childrenSubtreeQuotas;
        computeQuota(quota, subtreeQuota, childrenSubtreeQuotas, data);

        // Phase 3: Exchange particles
        exchangeWork(subtreeWork, subtreeQuota, childrenSubtreeQuotas, childrenSubtreeWork, pc);
    }

private:
    void reduceWorkLoad(int localparticles, int& subtreework, auto& childrensubtreework) {
        subtreework = localparticles;
        for (size_type i = 0; i < numChildren_m; ++i) {
            int temp;
            MPI_Status status;
            MPI_Recv(&temp, 1, MPI_INT, childrenRanks_m[i], childrenRanks_m[i], MPI_COMM_WORLD,
                     &status);
            subtreework += temp;
            childrensubtreework[i] = temp;
        }
        if (ippl::Comm->rank() != 0) {
            MPI_Request request;
            MPI_Isend(&subtreework, 1, MPI_INT, parentRank_m, ippl::Comm->rank(), MPI_COMM_WORLD,
                      &request);
        }
    }

    void computeQuota(int& quota, int& subtreeQuota, auto& childrenSubtreeQuotas, auto& data) {
        if (ippl::Comm->rank() < data[1]) {
            quota = data[0] + 1;
        } else {
            quota = data[0];
        }
        subtreeQuota = quota;

        if (numChildren_m > 0) {
            for (size_type i = 0; i < numChildren_m; ++i) {
                MPI_Status status;
                MPI_Recv(&childrenSubtreeQuotas[i], 1, MPI_INT, childrenRanks_m[i],
                         childrenRanks_m[i], MPI_COMM_WORLD, &status);
                subtreeQuota += childrenSubtreeQuotas[i];
            }
        }
        if (ippl::Comm->rank() != 0) {
            MPI_Request request;
            MPI_Isend(&subtreeQuota, 1, MPI_INT, parentRank_m, ippl::Comm->rank(), MPI_COMM_WORLD,
                      &request);
        }
        return;
    }

    template <class ParticleContainer>
    void exchangeWork(int& subtreeWork, int& subtreeQuota, auto& childrenSubtreeQuotas,
                      auto& childrenSubtreeWork, ParticleContainer& pc) {
        int tag = 0;

        if (debug_m) {
            std::cout << "Rank " << ippl::Comm->rank() << " exchangeWork: subtreeWork="
                      << subtreeWork << ", subtreeQuota=" << subtreeQuota << std::endl;
        }

        // Use same ordering as reference implementation
        // Receives happen first (blocking), then sends (non-blocking)
        // This works because sendToRank is non-blocking in IPPL

        if (subtreeWork < subtreeQuota && ippl::Comm->rank() != 0) {
            size_t nRecvs = subtreeQuota - subtreeWork;
            if (debug_m) {
                std::cout << "Rank " << ippl::Comm->rank() << " receiving " << nRecvs
                          << " particles from parent " << parentRank_m << std::endl;
            }
            pc.recvFromRank(parentRank_m, tag, nRecvs);
            if (debug_m) {
                std::cout << "Rank " << ippl::Comm->rank() << " received from parent" << std::endl;
            }
        }

        for (size_type i = 0; i < numChildren_m; ++i) {
            if (childrenSubtreeWork[i] > childrenSubtreeQuotas[i]) {
                size_t nRecvs = childrenSubtreeWork[i] - childrenSubtreeQuotas[i];
                if (debug_m) {
                    std::cout << "Rank " << ippl::Comm->rank() << " receiving " << nRecvs
                              << " particles from child " << childrenRanks_m[i] << std::endl;
                }
                pc.recvFromRank(childrenRanks_m[i], tag, nRecvs);
                if (debug_m) {
                    std::cout << "Rank " << ippl::Comm->rank() << " received from child "
                              << childrenRanks_m[i] << std::endl;
                }
            }
        }

        if (subtreeWork > subtreeQuota) {
            size_t numInvParticles = subtreeWork - subtreeQuota;
            if (debug_m) {
                std::cout << "Rank " << ippl::Comm->rank() << " sending " << numInvParticles
                          << " particles to parent " << parentRank_m << std::endl;
            }
            std::vector<MPI_Request> requests(0);
            hash_type hash("hash", numInvParticles);
            fillHash(numInvParticles, hash, pc);
            pc.sendToRank(parentRank_m, tag, requests, hash);
            bool_type invalidParticles("validity of particles", pc.getLocalNum());
            fillInvalid(numInvParticles, invalidParticles, pc);
            pc.internalDestroy(invalidParticles, numInvParticles);
            if (debug_m) {
                std::cout << "Rank " << ippl::Comm->rank() << " sent to parent" << std::endl;
            }
        }

        for (size_type i = 0; i < numChildren_m; ++i) {
            if (childrenSubtreeWork[i] < childrenSubtreeQuotas[i]) {
                size_t numInvParticles = childrenSubtreeQuotas[i] - childrenSubtreeWork[i];
                if (debug_m) {
                    std::cout << "Rank " << ippl::Comm->rank() << " sending " << numInvParticles
                              << " particles to child " << childrenRanks_m[i] << std::endl;
                }
                std::vector<MPI_Request> requests(0);
                hash_type hash("hash", numInvParticles);
                fillHash(numInvParticles, hash, pc);
                pc.sendToRank(childrenRanks_m[i], tag, requests, hash);
                bool_type invalidParticles("validity of particles", pc.getLocalNum());
                fillInvalid(numInvParticles, invalidParticles, pc);
                pc.internalDestroy(invalidParticles, numInvParticles);
                if (debug_m) {
                    std::cout << "Rank " << ippl::Comm->rank() << " sent to child "
                              << childrenRanks_m[i] << std::endl;
                }
            }
        }

        if (debug_m) {
            std::cout << "Rank " << ippl::Comm->rank() << " exchangeWork: DONE" << std::endl;
        }
    }

public:
    template <class ParticleContainer>
    void fillHash(int nParticles, hash_type& hash, ParticleContainer& pc) {
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_for(
            "ParticleTreeLayout::fillHash()",
            policy_type(pc.getLocalNum() - nParticles, pc.getLocalNum()),
            KOKKOS_LAMBDA(const size_t i) { hash(i - pc.getLocalNum() + nParticles) = i; });
    }

    template <class ParticleContainer>
    void fillInvalid(int nParticles, bool_type& invalidParticles, ParticleContainer& pc) {
        using policy_type = Kokkos::RangePolicy<position_execution_space>;
        Kokkos::parallel_for("ParticleTreeLayout::fillInvalid()", policy_type(0, pc.getLocalNum()),
                             KOKKOS_LAMBDA(const size_t i) {
                                 bool isInvalid       = i >= pc.getLocalNum() - nParticles;
                                 invalidParticles(i) = 0 + 1 * isInvalid;
                             });
    }

private:
    size_type parentRank_m;
    std::array<size_type, 3> childrenRanks_m;
    size_type numChildren_m;
    bool debug_m;
};

// ============================================================================
// IndependentParticles: Simple particle container without fields
// ============================================================================
template <class PLayout, typename T, unsigned Dim = 3>
class IndependentParticles : public ippl::ParticleBase<PLayout> {
    using Base = ippl::ParticleBase<PLayout>;

public:
    ParticleAttrib<Vector_t<T, Dim>> P;  // Velocity
    ParticleAttrib<T> q;                 // Charge

    IndependentParticles(PLayout& pl)
        : Base(pl) {
        this->addAttribute(P);
        this->addAttribute(q);
    }
};

#endif  // PARTICLE_TREE_LAYOUT_HPP
