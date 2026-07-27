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
#include <kamping/named_parameters.hpp>
#include <kamping/plugin/plugin_helpers.hpp>

#include "mpi/alltoallv/direct.hpp"
#include "mpi/alltoallv/native.hpp"
#include "mpi/alltoallv/onefactor.hpp"
#include "mpi/alltoallv/pairwise.hpp"
#include "mpi/alltoallv/params.hpp"

namespace dss_mehnert {
namespace mpi {

template <auto>
inline constexpr bool always_false_v = false;

template <typename Comm, template <typename...> typename DefaultContainerType>
class AlltoallvCombinedPlugin
    : public kamping::plugin::PluginBase<Comm, DefaultContainerType, AlltoallvCombinedPlugin> {
public:
    template <AlltoallvCombinedKind combined_type, typename SendBuf>
    auto alltoallv_combined(
        SendBuf&& send_buf,
        std::span<size_t const> send_counts,
        OneFactorParams const& onefactor_params = {}
    ) const {
        auto const recv_counts = this->to_communicator().alltoall(kamping::send_buf(send_counts));
        return alltoallv_combined<combined_type, SendBuf>(
            std::forward<SendBuf>(send_buf),
            send_counts,
            recv_counts,
            onefactor_params
        );
    }

    // `onefactor_params` only affects the one_factor kind; other kinds ignore it.
    template <AlltoallvCombinedKind kind, typename SendBuf>
    auto alltoallv_combined(
        SendBuf&& send_buf,
        std::span<size_t const> send_counts,
        std::span<size_t const> recv_counts,
        [[maybe_unused]] OneFactorParams const& onefactor_params = {}
    ) const {
        auto const& comm = this->to_communicator();

        if constexpr (kind == AlltoallvCombinedKind::combined) {
            auto const send_total =
                std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
            auto const recv_total =
                std::accumulate(recv_counts.begin(), recv_counts.end(), size_t{0});
            auto const local_max = std::max<size_t>(send_total, recv_total);
            auto const global_max = comm.allreduce_single(
                kamping::send_buf(local_max),
                kamping::op(kamping::ops::max<>{})
            );

            if (global_max < std::numeric_limits<int>::max()) {
                return alltoallv_native(comm, send_buf, send_counts, recv_counts);
            } else {
                return alltoallv_direct(comm, send_buf, send_counts, recv_counts);
            }
        } else if constexpr (kind == AlltoallvCombinedKind::native) {
            return alltoallv_native(comm, send_buf, send_counts, recv_counts);
        } else if constexpr (kind == AlltoallvCombinedKind::direct) {
            return alltoallv_direct(comm, send_buf, send_counts, recv_counts);
        } else if constexpr (kind == AlltoallvCombinedKind::one_factor) {
            return alltoallv_onefactor(comm, send_buf, send_counts, recv_counts, onefactor_params);
        } else if constexpr (kind == AlltoallvCombinedKind::pairwise) {
            return alltoallv_pairwise(comm, send_buf, send_counts, recv_counts);
        } else {
            []<AlltoallvCombinedKind type_ = kind> {
                static_assert(always_false_v<type_>, "invalid alltoallv combined kind used");
            }
            ();
        }
    }
};

} // namespace mpi
} // namespace dss_mehnert
