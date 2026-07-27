// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// The built-in MPI_Alltoallv, reached through kamping. Counts and displacements are
// `int`, so both the individual counts and the totals have to fit into an int32.

#pragma once

#include <algorithm>
#include <cstddef>
#include <span>
#include <utility>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>
#include <kassert/kassert.hpp>

namespace dss_mehnert {
namespace mpi {

template <typename Communicator, typename SendBuf>
auto alltoallv_native(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts
) {
    KAMPING_ASSERT(
        std::all_of(send_counts.begin(), send_counts.end(), std::in_range<int, size_t>),
        "all send counts need to fit into an int",
        kamping::assert::normal
    );
    KAMPING_ASSERT(
        std::all_of(recv_counts.begin(), recv_counts.end(), std::in_range<int, size_t>),
        "all recv counts need to fit into an int",
        kamping::assert::normal
    );

    std::vector<int> send_counts_int{send_counts.begin(), send_counts.end()};
    std::vector<int> recv_counts_int{recv_counts.begin(), recv_counts.end()};
    return comm.alltoallv(
        kamping::send_buf(send_buf),
        kamping::send_counts(send_counts_int),
        kamping::recv_counts(recv_counts_int)
    );
}

} // namespace mpi
} // namespace dss_mehnert
