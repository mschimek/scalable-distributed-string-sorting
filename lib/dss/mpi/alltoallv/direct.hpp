// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Direct point-to-point exchange using derived "big" datatypes, so that messages whose
// element count exceeds the int32 limit can still be sent. This is the fallback used when
// the counts are too large for the other implementations.

#pragma once

#include <cstddef>
#include <numeric>
#include <span>
#include <type_traits>
#include <vector>

#include <mpi.h>

#include "dss/mpi/alltoallv/log.hpp"
#include "dss/mpi/big_type.hpp"
#include "dss/util/measuringTool.hpp"

namespace dss_mehnert {
namespace mpi {

template <typename Communicator, typename SendBuf>
auto alltoallv_direct(
    Communicator const& comm,
    SendBuf&& send_buf,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts
) {
    using DataType = std::remove_reference_t<SendBuf>::value_type;

    _internal::log_alltoallv_impl("direct", comm.size());

    // todo this should use kamping, once irecv is merged into main
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    std::vector<size_t> send_displs(comm.size()), recv_displs(comm.size());
    std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
    std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

    auto const send_total = send_displs.back() + send_counts.back();
    measuring_tool.addRawCommunication(send_total * sizeof(DataType), "alltoallv");

    auto const recv_total = recv_displs.back() + recv_counts.back();
    std::vector<DataType> receive_data(recv_total);
    std::vector<MPI_Request> requests;
    requests.reserve(2 * comm.size());

    for (int i = 0; i < comm.size_signed(); ++i) {
        int source = (comm.rank_signed() + (comm.size_signed() - i)) % comm.size_signed();
        if (recv_counts[source] > 0) {
            auto receive_type = dss_schimek::mpi::get_big_type<DataType>(recv_counts[source]);
            MPI_Irecv(
                receive_data.data() + recv_displs[source],
                1,
                receive_type,
                source,
                44227,
                comm.mpi_communicator(),
                &requests.emplace_back(MPI_REQUEST_NULL)
            );
        }
    }
    for (int i = 0; i < comm.size_signed(); ++i) {
        int target = (comm.rank_signed() + i) % comm.size_signed();
        if (send_counts[target] > 0) {
            auto send_type = dss_schimek::mpi::get_big_type<DataType>(send_counts[target]);
            MPI_Issend(
                send_buf.data() + send_displs[target],
                1,
                send_type,
                target,
                44227,
                comm.mpi_communicator(),
                &requests.emplace_back(MPI_REQUEST_NULL)
            );
        }
    }
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
    return receive_data;
}

} // namespace mpi
} // namespace dss_mehnert
