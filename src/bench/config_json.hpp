// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Builds the `config` object written alongside the kamping timer JSON (--timer-json-path),
// documenting the full argument set a run was invoked with.

#pragma once

#include <cstddef>

#include <nlohmann/json.hpp>

#include "bench/args.hpp"
#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

nlohmann::ordered_json make_config_json(
    SorterArgs const& args,
    Communicator const& comm,
    std::size_t num_levels,
    std::size_t cpus_per_node
);

} // namespace bench
} // namespace dss_mehnert
