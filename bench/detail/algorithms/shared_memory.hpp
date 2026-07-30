// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <memory>

#include "detail/algorithm.hpp"
#include "detail/args.hpp"
#include "dss/mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

// Shared-memory baseline: tlx's parallel sample sort on a single PE. Reports its own timing
// rather than going through the MeasuringTool.
std::unique_ptr<AbstractAlgorithm>
make_shared_memory(SorterArgs const& args, Communicator const& comm);

} // namespace bench
} // namespace dss_mehnert
