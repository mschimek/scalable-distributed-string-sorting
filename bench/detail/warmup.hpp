// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Irregular alltoallv traffic run before a measured sort, so that the first exchange of the run
// does not pay for lazily established MPI connections.

#pragma once

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <tlx/die/core.hpp>

#include "dss/mpi/communicator.hpp"

namespace dss_mehnert {
namespace bench {

// Every PE sends a different, random number of bytes to every other PE, drawn uniformly from
// [min_bytes, max_bytes).
inline size_t
mpi_irregular_warmup(size_t const min_bytes, size_t const max_bytes, Communicator const& comm) {
    tlx_die_unless(min_bytes < max_bytes);

    std::mt19937_64 gen{comm.rank()};
    // uniform_int_distribution is inclusive on both ends, so draw from [min_bytes, max_bytes - 1]
    std::uniform_int_distribution<size_t> count_dist{min_bytes, max_bytes - 1};
    std::uniform_int_distribution<unsigned char> byte_dist{'A', 'Z'};

    std::vector<int> send_counts(comm.size());
    std::generate(send_counts.begin(), send_counts.end(), [&] {
        return static_cast<int>(count_dist(gen));
    });

    size_t const send_total = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
    std::vector<unsigned char> send_data(send_total);
    std::generate(send_data.begin(), send_data.end(), [&] { return byte_dist(gen); });

    // recv_counts are exchanged internally by kamping from the send_counts
    auto recv_data =
        comm.alltoallv(kamping::send_buf(send_data), kamping::send_counts(send_counts));

    auto volatile sum = std::accumulate(recv_data.begin(), recv_data.end(), size_t{0});
    return sum;
}

} // namespace bench
} // namespace dss_mehnert
