// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <string_view>
#include <utility>

#include <kamping/assertion_levels.hpp>
#include <kamping/communicator.hpp>
#include <kamping/named_parameter_check.hpp>
#include <kamping/named_parameter_selection.hpp>
#include <kamping/named_parameter_types.hpp>

#include "mpi/alltoall_combined.hpp"
#include "mpi/alltoall_strings.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace mpi {

template <
    template <typename...>
    typename DefaultContainerType,
    template <typename>
    typename... Plugins>
class TrackingCommunicator
    : public kamping::Communicator<DefaultContainerType>,
      public Plugins<TrackingCommunicator<DefaultContainerType, Plugins...>>... {
private:
    using Base = kamping::Communicator<DefaultContainerType>;

public:
    TrackingCommunicator() = default;

    TrackingCommunicator(Base comm) : Base(std::move(comm)) {}

#define DSS_MPI_TRACKING_DELEGATE(name)                       \
    template <typename... Args>                               \
    auto name(Args... args) const {                           \
        add_send_buf_size(#name, args...);                    \
        return this->Base::name(std::forward<Args>(args)...); \
    }

    DSS_MPI_TRACKING_DELEGATE(gather)
    DSS_MPI_TRACKING_DELEGATE(alltoall)

    DSS_MPI_TRACKING_DELEGATE(send)
    DSS_MPI_TRACKING_DELEGATE(bsend)
    DSS_MPI_TRACKING_DELEGATE(ssend)
    DSS_MPI_TRACKING_DELEGATE(rsend)
    DSS_MPI_TRACKING_DELEGATE(isend)
    DSS_MPI_TRACKING_DELEGATE(ibsend)
    DSS_MPI_TRACKING_DELEGATE(issend)
    DSS_MPI_TRACKING_DELEGATE(irsend)

    // todo is the sum of send_buf sizes correct here?
    DSS_MPI_TRACKING_DELEGATE(reduce)
    DSS_MPI_TRACKING_DELEGATE(allreduce)
    DSS_MPI_TRACKING_DELEGATE(scan)
    DSS_MPI_TRACKING_DELEGATE(scan_single)
    DSS_MPI_TRACKING_DELEGATE(exscan)
    DSS_MPI_TRACKING_DELEGATE(exscan_single)

#undef DSS_MPI_TRACKING_DELEGATE

    template <typename... Args>
    auto alltoallv(Args... args) const {
        add_send_recv_count_size<false>("alltoallv", args...);
        add_send_buf_size("alltoallv", args...);
        return this->Base::alltoallv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto allgather(Args... args) const {
        add_send_buf_size<false, true>("allgather", args...);
        return this->Base::allgatherv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto allgatherv(Args... args) const {
        add_send_recv_count_size<false>("allgatherv", args...);
        add_send_buf_size<false, true>("allgatherv", args...);
        return this->Base::allgatherv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto gatherv(Args... args) const {
        add_send_recv_count_size("gatherv", args...);
        add_send_buf_size("gatherv", args...);
        return this->Base::gatherv(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto scatter(Args... args) const {
        add_send_recv_count_size("scatter", args...);
        add_send_buf_size<true>("scatter", args...);
        return this->Base::scatter(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto scatterv(Args... args) const {
        namespace kmpi = kamping::internal;

        constexpr bool has_send_counts = kmpi::has_parameter_type<send_counts_t, Args...>();
        constexpr bool has_recv_counts = kmpi::has_parameter_type<recv_counts_t, Args...>();

        if constexpr (!has_send_counts && !has_recv_counts) {
            if (is_root_with_args(args...)) {
                add_comm_volume(this->size() * sizeof(int), "scatterv");
            }
        }

        add_send_buf_size<true>("scatterv", args...);
        return this->Base::scatter(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto bcast(Args... args) const {
        add_bcast_send_buf_size("bcast", args...);
        return this->Base::bcast(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto bcast_single(Args... args) const {
        add_bcast_send_buf_size("bcast_single", args...);
        return this->Base::bcast_single(std::forward<Args>(args)...);
    }

private:
    static constexpr auto root_t = kamping::internal::ParameterType::root;
    static constexpr auto send_buf_t = kamping::internal::ParameterType::send_buf;
    static constexpr auto send_recv_buf_t = kamping::internal::ParameterType::send_recv_buf;
    static constexpr auto send_counts_t = kamping::internal::ParameterType::send_counts;
    static constexpr auto recv_counts_t = kamping::internal::ParameterType::recv_counts;

    template <typename... Args>
    bool is_root_with_args(Args const&... args) const {
        namespace kmpi = kamping::internal;
        if constexpr (kmpi::has_parameter_type<root_t, Args...>()) {
            auto const& root_param = kmpi::select_parameter_type<root_t>(args...);
            return this->is_root(root_param.rank_signed());
        } else {
            return this->is_root();
        }
    }

    template <bool root_only = false, bool do_multiply = false, typename... Args>
    void add_send_buf_size(std::string_view description, Args&... args) const {
        namespace kmpi = kamping::internal;

        if constexpr (kmpi::has_parameter_type<send_buf_t, Args...>()) {
            if (!root_only || is_root_with_args(args...)) {
                auto const& send_buf = kmpi::select_parameter_type<send_buf_t>(args...);
                using send_value_type =
                    typename std::remove_reference_t<decltype(send_buf)>::value_type;

                auto send_buf_size = send_buf.size() * sizeof(send_value_type);
                auto total_size = do_multiply ? send_buf_size : send_buf_size * this->size();
                add_comm_volume(total_size, description);
            }
        }
    }

    template <typename... Args>
    void add_bcast_send_buf_size(std::string_view description, Args const&... args) const {
        namespace kmpi = kamping::internal;

        if constexpr (kmpi::has_parameter_type<send_recv_buf_t, Args...>()) {
            if (is_root_with_args(args...)) {
                auto const& send_buf = kmpi::select_parameter_type<send_recv_buf_t>(args...);
                using send_value_type =
                    typename std::remove_reference_t<decltype(send_buf)>::value_type;

                auto send_buf_size = send_buf.size() * sizeof(send_value_type);
                add_comm_volume(this->size() * send_buf_size, description);
            }
        }
    }

    template <bool root_only = true, typename... Args>
    void add_send_recv_count_size(std::string_view description, Args const&... args) const {
        namespace kmpi = kamping::internal;

        if constexpr (!kmpi::has_parameter_type<recv_counts_t, Args...>()) {
            if (!root_only || is_root_with_args(args...)) {
                add_comm_volume(this->size() * sizeof(int), description);
            }
        }
    }

    void add_comm_volume(size_t volume_bytes, std::string_view description) const {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();
        measuring_tool.addRawCommunication(volume_bytes, description);
    }
};

using Communicator =
    TrackingCommunicator<std::vector, mpi::AlltoallvCombinedPlugin, mpi::AlltoallStringsPlugin>;

} // namespace mpi

using mpi::Communicator;

} // namespace dss_mehnert
