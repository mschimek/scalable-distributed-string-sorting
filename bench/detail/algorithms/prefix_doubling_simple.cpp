// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Prefix doubling with the simple permutation.

#include "detail/algorithms/prefix_doubling_impl.hpp"

namespace dss_mehnert {
namespace bench {

std::unique_ptr<AbstractAlgorithm>
make_prefix_doubling_simple(SorterArgs const& args, Communicator const& comm) {
    return make_prefix_doubling<SimplePermutation>(args, comm);
}

} // namespace bench
} // namespace dss_mehnert
