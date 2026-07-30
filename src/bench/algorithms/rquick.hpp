// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <memory>

#include "bench/algorithm.hpp"
#include "executables/args.hpp"
#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

// The LCP-aware hypercube quicksort (RQuick2) as the top-level sorter. `--rquick-lcp` selects
// whether it sorts with LCP values.
std::unique_ptr<AbstractAlgorithm> make_rquick(SorterArgs const& args, Communicator const& comm);

} // namespace bench
} // namespace dss_mehnert
