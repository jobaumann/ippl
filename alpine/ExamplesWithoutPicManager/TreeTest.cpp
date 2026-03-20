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

    std::cout << prefix << (isLast ? "└── " : "├── ")
              << "rank " << node->getOrder()
              << "  [depth=" << node->getDepth()
              << ", leaves_to_dist=" << node->getNumLeaves()
              << "]" << std::endl;

    auto children = node->getChildren();
    size_type nch = node->getNumChildren();
    std::string childPrefix = prefix + (isLast ? "    " : "│   ");
    for (size_type i = 0; i < nch; ++i)
        printTree(children[i], childPrefix, i == nch - 1);
}

// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        int nRanks = ippl::Comm->size();
        int myRank = ippl::Comm->rank();

        // Construct Tree exactly ONCE on every rank.
        // determineOrder() uses a static counter — constructing it a second time
        // on any rank would produce wrong parent/children data.
        Tree t(nRanks);

        MPI_Barrier(MPI_COMM_WORLD);

        // ── Rank 0: ASCII tree + Graphviz ─────────────────────────────────────
        if (myRank == 0) {
            std::cout << "\n=============================" << std::endl;
            std::cout << " Load-Balancing Tree (" << nRanks << " ranks)" << std::endl;
            std::cout << "=============================" << std::endl;

            auto root = t.getRoot();
            std::cout << "\nrank " << root->getOrder()
                      << "  [depth=" << root->getDepth()
                      << ", leaves_to_dist=" << root->getNumLeaves()
                      << "]  (root)" << std::endl;
            auto children = root->getChildren();
            size_type nch = root->getNumChildren();
            for (size_type i = 0; i < nch; ++i)
                printTree(children[i], "", i == nch - 1);

            std::cout << std::endl;
            t.generateGraphvizOutput(root, "tree_structure.dot");
            std::cout << "Render with: dot -Tpng tree_structure.dot -o tree.png" << std::endl;
            std::cout << "=============================" << std::endl;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        // ── All ranks: print own position in tree in order ────────────────────
        if (myRank == 0) {
            std::cout << "\nPer-rank summary:" << std::endl;
            std::cout << "-----------------------------" << std::endl;
        }
        MPI_Barrier(MPI_COMM_WORLD);

        for (int r = 0; r < nRanks; ++r) {
            if (myRank == r) {
                std::cout << "Rank " << r << ": ";
                if (r == 0)
                    std::cout << "parent=none (root)";
                else
                    std::cout << "parent=" << t.getGlobalParent();
                std::cout << ", numChildren=" << t.getGlobalNumChildren() << ", children=[";
                auto ch = t.getGlobalChildren();
                for (size_type i = 0; i < t.getGlobalNumChildren(); ++i) {
                    std::cout << ch[i];
                    if (i + 1 < t.getGlobalNumChildren()) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                std::cout << std::flush;
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }
    }
    ippl::finalize();
    return 0;
}
