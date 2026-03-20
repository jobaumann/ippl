// TreeTest: Print the load-balancing communication tree for the current MPI config.
//
// Usage:
//   mpirun -n <N> ./TreeTest
//
// Output:
//   - ASCII tree printed by rank 0
//   - Per-rank parent/children summary
//   - tree_structure.dot (render with: dot -Tpng tree_structure.dot -o tree.png)

#include <iostream>
#include <memory>
#include <string>
#include "Ippl.h"
#include "ParticleTreeLayout.hpp"

// ── ASCII tree printer ────────────────────────────────────────────────────────
void printTree(std::shared_ptr<Node<size_type>> node,
               const std::string& prefix = "",
               bool isLast = true) {
    if (!node) return;

    std::cout << prefix;
    std::cout << (isLast ? "└── " : "├── ");
    std::cout << "rank " << node->getOrder()
              << "  [depth=" << node->getDepth()
              << ", leaves_to_dist=" << node->getNumLeaves()
              << "]" << std::endl;

    auto children  = node->getChildren();
    size_type nch  = node->getNumChildren();
    std::string childPrefix = prefix + (isLast ? "    " : "│   ");
    for (size_type i = 0; i < nch; ++i) {
        printTree(children[i], childPrefix, i == nch - 1);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        int nRanks = ippl::Comm->size();
        int myRank = ippl::Comm->rank();

        MPI_Barrier(MPI_COMM_WORLD);

        if (myRank == 0) {
            std::cout << "\n=============================" << std::endl;
            std::cout << " Load-Balancing Tree (" << nRanks << " ranks)" << std::endl;
            std::cout << "=============================" << std::endl;

            Tree t(nRanks);

            // ASCII tree
            std::cout << "\nrank " << t.getRoot()->getOrder()
                      << "  [depth=" << t.getRoot()->getDepth()
                      << ", leaves_to_dist=" << t.getRoot()->getNumLeaves()
                      << "]  (root)" << std::endl;
            auto children = t.getRoot()->getChildren();
            size_type nch = t.getRoot()->getNumChildren();
            for (size_type i = 0; i < nch; ++i) {
                printTree(children[i], "", i == nch - 1);
            }

            // Graphviz file
            std::cout << std::endl;
            t.generateGraphvizOutput(t.getRoot(), "tree_structure.dot");
            std::cout << "Render with: dot -Tpng tree_structure.dot -o tree.png" << std::endl;
            std::cout << "=============================" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // Each rank prints its own position using ParticleTreeLayout
        // (debug=true triggers the per-rank parent/children printout)
        std::cout << std::flush;
        MPI_Barrier(MPI_COMM_WORLD);

        if (myRank == 0) {
            std::cout << "\nPer-rank summary:" << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        ParticleTreeLayout<double, 3> layout(/*debug=*/true);
    }
    ippl::finalize();
    return 0;
}
