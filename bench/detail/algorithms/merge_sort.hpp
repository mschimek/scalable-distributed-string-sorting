// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <memory>

#include "detail/algorithm.hpp"
#include "detail/args.hpp"
#include "dss/mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

// Multi-level distributed merge sort. Defined in merge_sort.cpp, which is the only place its
// template tree is instantiated.
std::unique_ptr<AbstractAlgorithm>
make_merge_sort(SorterArgs const& args, Communicator const& comm);

} // namespace bench
} // namespace dss_mehnert
