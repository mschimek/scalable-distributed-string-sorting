// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Prefix doubling with the multi-level permutation.

#include "detail/algorithms/prefix_doubling_impl.hpp"

namespace dss_mehnert {
namespace bench {

std::unique_ptr<AbstractAlgorithm>
make_prefix_doubling_multi_level(SorterArgs const& args, Communicator const& comm) {
    return make_prefix_doubling<MultiLevelPermutation>(args, comm);
}

} // namespace bench
} // namespace dss_mehnert
