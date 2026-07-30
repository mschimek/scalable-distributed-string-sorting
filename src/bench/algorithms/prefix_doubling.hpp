// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <memory>

#include "bench/algorithm.hpp"
#include "bench/args.hpp"
#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

// Prefix-doubling merge sort, which returns a permutation rather than the strings themselves.
// The permutation type is the axis the implementation is split along: one translation unit per
// permutation, each including prefix_doubling_impl.hpp.
std::unique_ptr<AbstractAlgorithm>
make_prefix_doubling_simple(SorterArgs const& args, Communicator const& comm);

std::unique_ptr<AbstractAlgorithm>
make_prefix_doubling_multi_level(SorterArgs const& args, Communicator const& comm);

} // namespace bench
} // namespace dss_mehnert
