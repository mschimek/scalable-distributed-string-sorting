// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Pairwise MPI_Alltoallv: a reimplementation of Open MPI's
// ompi_coll_base_alltoallv_intra_pairwise using public (kamping) point-to-point calls.

#pragma once

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include <kamping/named_parameters.hpp>
#include <kamping/p2p/irecv.hpp>
#include <kamping/p2p/send.hpp>
#include <kamping/request.hpp>

#include "dss/mpi/alltoallv/log.hpp"

namespace dss_mehnert {
namespace mpi {

// Pairwise exchange, a reimplementation of Open MPI's ompi_coll_base_alltoallv_intra_pairwise. At
// step `s` every PE sends to (rank + s) and receives from (rank - s).
template <typename Communicator, typename SendBuf>
auto alltoallv_pairwise(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts
) {
    using namespace kamping;
    using DataType = std::remove_reference_t<SendBuf>::value_type;

    _internal::log_alltoallv_impl("pairwise", comm.size());

    auto const p = static_cast<size_t>(comm.size());

    std::vector<size_t> send_displs(p), recv_displs(p);
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    // comm volume is tracked by TrackingCommunicator on the comm.send below
    auto const recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType> receive_data(recv_total);

    int const size = comm.size_signed();
    int const rank = comm.rank_signed();
    static constexpr int msg_tag = 44228;

    for (int step = 0; step < size; ++step) {
        int const send_to = (rank + step) % size;
        int const recv_from = (rank + size - step) % size;

        // post the receive before the blocking send; this also makes the step-0 self-exchange safe,
        // since the matching receive is already in place
        Request request;
        if (recv_counts[recv_from] > 0) {
            std::span<DataType> rbuf{
                receive_data.data() + recv_displs[recv_from],
                recv_counts[recv_from]
            };
            comm.irecv(
                kamping::recv_buf(rbuf),
                kamping::recv_count(static_cast<int>(recv_counts[recv_from])),
                source(recv_from),
                tag(msg_tag),
                kamping::request(request)
            );
        }
        if (send_counts[send_to] > 0) {
            std::span<DataType const> sbuf{
                send_buf.data() + send_displs[send_to],
                send_counts[send_to]
            };
            comm.send(kamping::send_buf(sbuf), destination(send_to), tag(msg_tag));
        }
        if (recv_counts[recv_from] > 0) {
            request.wait();
        }
    }

    return receive_data;
}

} // namespace mpi
} // namespace dss_mehnert
