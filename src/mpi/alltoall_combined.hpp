// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameter_check.hpp>
#include <kamping/named_parameter_selection.hpp>
#include <kamping/named_parameter_types.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die/core.hpp>

#include "mpi/big_type.hpp"
#include "mpi/plugin_helpers.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace mpi {

enum class AlltoallvCombinedKind { combined, native, direct };

template <typename Comm>
class AlltoallvCombinedPlugin : public kamping::plugins::PluginBase<Comm, AlltoallvCombinedPlugin> {
public:
    template <AlltoallvCombinedKind combined_type, typename SendBuf>
    auto alltoallv_combined(SendBuf&& send_buf, std::span<size_t const> send_counts) const {
        auto const recv_counts =
            this->to_communicator().alltoall(kamping::send_buf(send_counts)).extract_recv_buffer();
        return alltoallv_combined<combined_type, SendBuf>(
            std::forward<SendBuf>(send_buf),
            send_counts,
            recv_counts
        );
    }

    template <AlltoallvCombinedKind kind, typename SendBuf>
    auto alltoallv_combined(
        SendBuf&& send_buf, std::span<size_t const> send_counts, std::span<size_t const> recv_counts
    ) const {
        if constexpr (kind == AlltoallvCombinedKind::combined) {
            auto const send_total =
                std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
            auto const recv_total =
                std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0});
            auto const local_max = std::max<size_t>(send_total, recv_total);
            auto const global_max = this->to_communicator().allreduce_single(
                kamping::send_buf(local_max),
                kamping::op(kamping::ops::max<>{})
            );

            if (global_max < std::numeric_limits<int>::max()) {
                return alltoallv_native(send_buf, send_counts, recv_counts);
            } else {
                return alltoallv_direct(send_buf, send_counts, recv_counts);
            }
        } else if constexpr (kind == AlltoallvCombinedKind::native) {
            return alltoallv_native(send_buf, send_counts, recv_counts);
        } else if constexpr (kind == AlltoallvCombinedKind::direct) {
            return alltoallv_direct(send_buf, send_counts, recv_counts);
        } else {
            []<AlltoallvCombinedKind type_ = kind> {
                static_assert(type_ != type_, "invalid alltoallv combined kind used");
            }
            ();
        }
    }

private:
    template <typename SendBuf>
    auto alltoallv_native(
        SendBuf&& send_buf, std::span<size_t const> send_counts, std::span<size_t const> recv_counts
    ) const {
        KASSERT(
            std::all_of(send_counts.begin(), send_counts.end(), std::in_range<int, size_t>),
            "all send counts need to fit into an int",
            kamping::assert::normal
        );
        KASSERT(
            std::all_of(recv_counts.begin(), recv_counts.end(), std::in_range<int, size_t>),
            "all recv counts need to fit into an int",
            kamping::assert::normal
        );

        std::vector<int> send_counts_int{send_counts.begin(), send_counts.end()};
        std::vector<int> recv_counts_int{recv_counts.begin(), recv_counts.end()};
        return this->to_communicator()
            .alltoallv(
                kamping::send_buf(send_buf),
                kamping::send_counts(send_counts_int),
                kamping::recv_counts(recv_counts_int)
            )
            .extract_recv_buffer();
    }

    template <typename SendBuf>
    auto alltoallv_direct(
        SendBuf&& send_buf, std::span<size_t const> send_counts, std::span<size_t const> recv_counts
    ) const {
        using DataType = std::remove_reference_t<SendBuf>::value_type;

        // todo this should use kamping, once irecv is merged into main
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        auto const& comm = this->to_communicator();
        std::vector<size_t> send_displs(comm.size()), recv_displs(comm.size());
        std::exclusive_scan(send_counts.begin(), send_counts.end(), send_displs.begin(), size_t{0});
        std::exclusive_scan(recv_counts.begin(), recv_counts.end(), recv_displs.begin(), size_t{0});

        auto const send_total = send_displs.back() + send_counts.back();
        measuring_tool.addRawCommunication(send_total * sizeof(DataType), "alltoallv_direct");

        auto const recv_total = recv_displs.back() + recv_counts.back();
        std::vector<DataType> receive_data(recv_total);
        std::vector<MPI_Request> mpi_request(2 * comm.size());

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
                    &mpi_request[source]
                );
            }
        }
        for (int i = 0; i < comm.size_signed(); ++i) {
            int target = (comm.rank_signed() + i) % comm.size_signed();
            if (send_counts[target] > 0) {
                auto send_type = dss_schimek::mpi::get_big_type<DataType>(send_counts[target]);
                MPI_Isend(
                    send_buf.data() + send_displs[target],
                    1,
                    send_type,
                    target,
                    44227,
                    comm.mpi_communicator(),
                    &mpi_request[comm.size() + target]
                );
            }
        }
        MPI_Waitall(2 * comm.size(), mpi_request.data(), MPI_STATUSES_IGNORE);
        return receive_data;
    }
};

} // namespace mpi
} // namespace dss_mehnert
