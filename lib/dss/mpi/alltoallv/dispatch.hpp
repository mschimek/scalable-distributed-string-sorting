// (c) 2018 Florian Kurpicz
// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Selects between the alltoallv implementations in this directory and exposes them as a
// kamping communicator plugin.

#pragma once

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <utility>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/plugin/plugin_helpers.hpp>
#include <tlx/die/core.hpp>

#include "dss/mpi/alltoallv/direct.hpp"
#include "dss/mpi/alltoallv/native.hpp"
#include "dss/mpi/alltoallv/onefactor.hpp"
#include "dss/mpi/alltoallv/pairwise.hpp"
#include "dss/mpi/alltoallv/params.hpp"

namespace dss_mehnert {
namespace mpi {

namespace _internal {

// The largest quantity `algorithm` has to squeeze into an int32 for this exchange. The
// algorithms differ here: MPI_Alltoallv takes int displacements, so its *totals* must fit,
// whereas the point-to-point schedules only ever cast an individual count.
inline size_t max_int32_critical_count(
    AlltoallvAlgorithm const algorithm,
    std::span<size_t const> send_counts,
    std::span<size_t const> recv_counts
) {
    auto const max_of = [](std::span<size_t const> counts) {
        auto const it = std::max_element(counts.begin(), counts.end());
        return it == counts.end() ? size_t{0} : *it;
    };
    auto const total_of = [](std::span<size_t const> counts) {
        return std::accumulate(counts.begin(), counts.end(), size_t{0});
    };

    switch (algorithm) {
        case AlltoallvAlgorithm::native:
            // displacements are ints, so the totals have to fit as well
            return std::max(total_of(send_counts), total_of(recv_counts));
        case AlltoallvAlgorithm::onefactor:
        case AlltoallvAlgorithm::pairwise:
            // per-partner point-to-point, only the individual counts are cast to int
            return std::max(max_of(send_counts), max_of(recv_counts));
        case AlltoallvAlgorithm::direct:
            // uses derived big datatypes, no int32 limit applies
            return 0;
    }
    tlx_die("unknown alltoallv algorithm");
}

} // namespace _internal

template <typename Comm, template <typename...> typename DefaultContainerType>
class AlltoallvPlugin
    : public kamping::plugin::PluginBase<Comm, DefaultContainerType, AlltoallvPlugin> {
public:
    template <typename SendBuf>
    auto alltoallv_dispatch(
        SendBuf&& send_buf, std::span<size_t const> send_counts, AlltoallvParams const& params = {}
    ) const {
        auto const recv_counts = this->to_communicator().alltoall(kamping::send_buf(send_counts));
        return alltoallv_dispatch<SendBuf>(
            std::forward<SendBuf>(send_buf),
            send_counts,
            recv_counts,
            params
        );
    }

    template <typename SendBuf>
    auto alltoallv_dispatch(
        SendBuf&& send_buf,
        std::span<size_t const> send_counts,
        std::span<size_t const> recv_counts,
        AlltoallvParams const& params = {}
    ) const {
        auto const& comm = this->to_communicator();

        auto algorithm = params.algorithm;
        if (params.large_counts && algorithm != AlltoallvAlgorithm::direct) {
            auto const local_max =
                _internal::max_int32_critical_count(algorithm, send_counts, recv_counts);
            auto const global_max = comm.allreduce_single(
                kamping::send_buf(local_max),
                kamping::op(kamping::ops::max<>{})
            );

            if (global_max >= static_cast<size_t>(std::numeric_limits<int>::max())) {
                algorithm = AlltoallvAlgorithm::direct;
            }
        }

        // barrier on `comm`, not on MPI_COMM_WORLD as synchronize_and_start() would
        comm.barrier();
        kamping::measurements::timer().start("alltoallv");

        auto result = [&] {
            switch (algorithm) {
                case AlltoallvAlgorithm::native:
                    return alltoallv_native(comm, send_buf, send_counts, recv_counts);
                case AlltoallvAlgorithm::direct:
                    return alltoallv_direct(comm, send_buf, send_counts, recv_counts);
                case AlltoallvAlgorithm::onefactor:
                    return alltoallv_onefactor(
                        comm,
                        send_buf,
                        send_counts,
                        recv_counts,
                        params.onefactor
                    );
                case AlltoallvAlgorithm::pairwise:
                    return alltoallv_pairwise(comm, send_buf, send_counts, recv_counts);
            }
            tlx_die("unknown alltoallv algorithm");
        }();

        kamping::measurements::timer().stop_and_append();
        return result;
    }
};

} // namespace mpi
} // namespace dss_mehnert
