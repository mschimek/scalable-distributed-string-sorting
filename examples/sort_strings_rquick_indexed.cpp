// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

// Distributed string sorting with the indexed top-level RQuick (V2) sorter: each
// string carries a 64-bit index, and strings are ordered lexicographically with
// the index as a tie-breaker. This is the same indexed configuration RQuick2
// uses for splitter candidates. See sort_strings_rquick.cpp for the unindexed
// variant.

#include <kamping/environment.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/fmt/ranges.h>

#include "dss/dss.hpp"
#include "dss/mpi/communicator.hpp"
#include "example_common.hpp"

int main(int argc, char** argv) {
    kamping::Environment env{argc, argv};
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();
    dss_mehnert::Communicator comm{};

    auto const cfg = dss::examples::InputConfig{.strings_per_rank = 2, .seed = 0xBADCAFE};
    auto local_input = dss::examples::make_all_a_local_input(comm.rank(), cfg);
    auto local_indices = dss::examples::make_local_indices(comm.rank(), cfg);
    auto input_copy = local_input; // run_rquick consumes its arguments
    auto indices_copy = local_indices;
    SPDLOG_LOGGER_INFO(
        spdlog::get("gather"),
        "sorted strings {}",
        dss::examples::zip_nul(local_input, local_indices)
    );

    auto [sorted_local, sorted_indices] = dss::run_rquick(local_input, local_indices, comm);

    SPDLOG_LOGGER_INFO(
        spdlog::get("gather"),
        "sorted strings {}",
        dss::examples::zip_nul(sorted_local, sorted_indices)
    );

    return dss::examples::verify_and_report_indexed(
        comm,
        sorted_local,
        sorted_indices,
        input_copy,
        indices_copy,
        "example-rquick-indexed"
    );
}
