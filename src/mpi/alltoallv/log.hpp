// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Debug output naming the alltoallv implementation an exchange actually ran in. Reported by
// the implementations themselves rather than by the dispatcher, so that what is logged is
// the code that ran, not the choice that was made.

#pragma once

#include <cstddef>
#include <set>
#include <string_view>
#include <utility>

#include <kamping/communicator.hpp>
#include <spdlog/spdlog.h>

namespace dss_mehnert {
namespace mpi {
namespace _internal {

// The first exchange of each (implementation, communicator size) is announced once at info
// level, every exchange is logged at debug level (SPDLOG_LEVEL=debug, and only in a build whose
// SPDLOG_ACTIVE_LEVEL keeps the debug macros -- Release compiles them out). One line per call
// is far too much for a run to be readable by default, but one line per implementation is
// exactly the check that a run uses the variant it was configured with.
//
// World PE 0 only, and a no-op unless the application has set up the loggers.
inline void log_alltoallv_impl(std::string_view const name, size_t const comm_size) {
    // looked up once: the lookup takes a lock on the spdlog registry, and this sits on the
    // hot path of every exchange
    static auto const logger = spdlog::get("root");
    if (logger == nullptr || kamping::world_rank() != 0) {
        return;
    }

    static std::set<std::pair<std::string_view, size_t>> announced;
    if (announced.emplace(name, comm_size).second) {
        SPDLOG_LOGGER_INFO(logger, "alltoallv: running {} on {} PEs", name, comm_size);
    }
    SPDLOG_LOGGER_DEBUG(logger, "alltoallv: running {} on {} PEs", name, comm_size);
}

} // namespace _internal
} // namespace mpi
} // namespace dss_mehnert
