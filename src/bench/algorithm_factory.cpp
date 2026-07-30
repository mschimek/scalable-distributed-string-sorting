// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "bench/algorithm_factory.hpp"

#include <tlx/die/core.hpp>

#include "bench/algorithms/merge_sort.hpp"
#include "bench/algorithms/prefix_doubling.hpp"
#include "bench/algorithms/rquick.hpp"
#include "bench/algorithms/shared_memory.hpp"

namespace dss_mehnert {
namespace bench {

std::unique_ptr<AbstractAlgorithm>
make_algorithm(SorterArgs const& args, Communicator const& comm) {
    switch (args.algorithm) {
        case Algorithm::merge_sort: {
            return make_merge_sort(args, comm);
        }
        case Algorithm::prefix_doubling: {
            switch (args.permutation) {
                case Permutation::simple: {
                    return make_prefix_doubling_simple(args, comm);
                }
                case Permutation::multi_level: {
                    return make_prefix_doubling_multi_level(args, comm);
                }
                case Permutation::sentinel: {
                    break;
                }
            }
            tlx_die("invalid permutation");
        }
        case Algorithm::rquick: {
            return make_rquick(args, comm);
        }
        case Algorithm::shared_memory: {
            return make_shared_memory(args, comm);
        }
        case Algorithm::sentinel: {
            break;
        }
    }
    tlx_die("invalid algorithm");
}

} // namespace bench
} // namespace dss_mehnert
