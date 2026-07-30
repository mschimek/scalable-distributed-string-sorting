// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The single entry point the runner uses to construct the algorithm selected on the command
// line. Defined in algorithm_factory.cpp, which is the only translation unit that sees all four
// per-family make_* declarations at once; each of those is itself defined in its own TU (see
// bench/algorithms/), so this switch is the only place their template trees are ever adjacent.

#pragma once

#include <memory>

#include "bench/algorithm.hpp"
#include "bench/args.hpp"
#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

std::unique_ptr<AbstractAlgorithm>
make_algorithm(SorterArgs const& args, Communicator const& comm);

} // namespace bench
} // namespace dss_mehnert
