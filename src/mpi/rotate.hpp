// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <vector>

#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/recv.hpp>
#include <kamping/p2p/send.hpp>
#include <mpi.h>

#include "mpi/communicator.hpp"

namespace dss_mehnert {
namespace mpi {

template <typename CharType>
void rotate_strings_right(
    std::vector<CharType> const& source,
    std::vector<CharType>& dest,
    bool const skip_rank,
    Communicator const& comm
) {
    namespace kmp = kamping;
    int const tag = comm.default_tag();

    assert(!(skip_rank && comm.is_root()));

    // this has latency O(p * α) in the worst case
    int const pred = comm.rank_shifted_cyclic(-1);
    int const succ = comm.rank_shifted_cyclic(1);
    int send_count = source.size(), recv_count = 0;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    // clang-format off
    if (skip_rank) {
        comm.recv(kmp::recv_buf(send_count), kmp::recv_counts(1), kmp::source(pred), kmp::tag(tag));
        comm.send(kmp::send_buf(send_count), kmp::destination(succ), kmp::tag(tag));
        recv_count = send_count;
    } else {
        MPI_Sendrecv(&send_count, 1, kmp::mpi_datatype<int>(), succ, tag,
                     &recv_count, 1, kmp::mpi_datatype<int>(), pred, tag,
                     comm.mpi_communicator(), MPI_STATUS_IGNORE);
        measuring_tool.addRawCommunication(sizeof(int), "rotate_strings");
    }
    dest.resize(recv_count);

    if (skip_rank) {
        comm.recv(kmp::recv_buf(dest), kmp::recv_counts(recv_count), kmp::source(pred), kmp::tag(tag));
        comm.send(kmp::send_buf(dest), kmp::destination(succ), kmp::tag(tag));
    } else {
        MPI_Sendrecv(source.data(), send_count, kmp::mpi_datatype<CharType>(), succ, tag,
                     dest.data(), recv_count, kmp::mpi_datatype<CharType>(), pred, tag,
                     comm.mpi_communicator(), MPI_STATUS_IGNORE);
        measuring_tool.addRawCommunication(send_count * sizeof(CharType), "rotate_strings");
    }
    // clang-format on
}


} // namespace mpi
} // namespace dss_mehnert
