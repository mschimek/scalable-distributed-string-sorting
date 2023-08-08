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
#include <kamping/plugin_helpers.hpp>

#include "util/measuringTool.hpp"

namespace dss_mehnert {

/// @brief A plugin providing `allreduce_single`, a simple wrapper around `allreduce`.
template <typename Comm>
class AllReduceSinglePlugin : public kamping::plugins::PluginBase<Comm, AllReduceSinglePlugin> {
public:
    /// @brief Wrapper for \c kamping::Communicator::allreduce but restricted to a single element.
    template <typename... Args>
    auto allreduce_single(Args... args) const {
        using kamping::internal::ParameterType;

        KAMPING_CHECK_PARAMETERS(
            Args,
            KAMPING_REQUIRED_PARAMETERS(send_buf, op),
            KAMPING_OPTIONAL_PARAMETERS()
        );

        KASSERT(
            select_parameter_type<ParameterType::send_recv_buf>(args...).size() == 1u,
            "The send buffer has to be of size 1 on all ranks.",
            kamping::assert::light
        );

        return this->to_communicator()
            .allreduce(std::forward<Args>(args)...)
            .extract_recv_buffer()[0];
    }
};

// todo consider communication volume for things like recv counts
template <typename Comm>
class CommVolumePlugin : public kamping::plugins::PluginBase<Comm, CommVolumePlugin> {
public:
#define DSS_MPI_TRACKING_DELEGATE(name)                                   \
    template <typename... Args>                                           \
    auto name(Args... args) const {                                       \
        add_send_buf_size(#name, args...);                                \
        return this->to_communicator().name(std::forward<Args>(args)...); \
    }

    DSS_MPI_TRACKING_DELEGATE(scatter)
    DSS_MPI_TRACKING_DELEGATE(scatterv)

    DSS_MPI_TRACKING_DELEGATE(gather)
    DSS_MPI_TRACKING_DELEGATE(gatherv)

    DSS_MPI_TRACKING_DELEGATE(alltoall)
    DSS_MPI_TRACKING_DELEGATE(alltoallv)

    DSS_MPI_TRACKING_DELEGATE(allgather)
    DSS_MPI_TRACKING_DELEGATE(allgatherv)

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
    DSS_MPI_TRACKING_DELEGATE(escan_single)

#undef DSS_MPI_TRACKING_DELEGATE

    template <typename... Args>
    auto bcast(Args... args) const {
        add_bcast_send_buf_size("bcast", args...);
        return this->to_communicator().bcast(std::forward<Args>(args)...);
    }

    template <typename... Args>
    auto bcast_single(Args... args) const {
        add_bcast_send_buf_size("bcast_single", args...);
        return this->to_communicator().bcast_single(std::forward<Args>(args)...);
    }

private:
    static constexpr auto send_buf_t = kamping::internal::ParameterType::send_buf;
    static constexpr auto send_recv_buf_t = kamping::internal::ParameterType::send_recv_buf;
    static constexpr auto root_t = kamping::internal::ParameterType::root;

    template <typename... Args>
    void add_send_buf_size(std::string_view description, Args&... args) const {
        namespace kmp = kamping::internal;

        if constexpr (kmp::has_parameter_type<send_buf_t, Args...>()) {
            auto const& send_buf = kmp::select_parameter_type<send_buf_t, Args...>(args...);
            using send_value_type =
                typename std::remove_reference_t<decltype(send_buf)>::value_type;

            auto send_buf_size = send_buf.size() * sizeof(send_value_type);

            auto& measuring_tool = measurement::MeasuringTool::measuringTool();
            measuring_tool.addRawCommunication(send_buf_size, description);
        }
    }

    template <typename... Args>
    void add_bcast_send_buf_size(std::string_view description, Args const&... args) const {
        namespace kmp = kamping::internal;

        if constexpr (kmp::has_parameter_type<send_recv_buf_t, Args...>()) {
            auto const& comm = this->to_communicator();

            bool is_root;
            if constexpr (kmp::has_parameter_type<root_t>()) {
                size_t root = kmp::select_parameter_type<root_t>(args...);
                is_root = comm.is_root(root);
            } else {
                is_root = comm.is_root();
            }

            if (is_root) {
                auto const& send_buf = kmp::select_parameter_type<send_recv_buf_t>(args...);
                using send_value_type =
                    typename std::remove_reference_t<decltype(send_buf)>::value_type;


                auto send_buf_size = send_buf.size() * sizeof(send_value_type);
                auto total_size = comm.size() * send_buf_size;

                auto& measuring_tool = measurement::MeasuringTool::measuringTool();
                measuring_tool.addRawCommunication(total_size, description);
            }
        }
    }
};

class TrackingCommunicator
    : public kamping::Communicator<std::vector, AllReduceSinglePlugin, CommVolumePlugin> {
public:
    using CommVolumePlugin::allgather;
    using CommVolumePlugin::allgatherv;
    using CommVolumePlugin::allreduce;
    using CommVolumePlugin::alltoall;
    using CommVolumePlugin::alltoallv;
    using CommVolumePlugin::bsend;
    using CommVolumePlugin::escan_single;
    using CommVolumePlugin::exscan;
    using CommVolumePlugin::gather;
    using CommVolumePlugin::gatherv;
    using CommVolumePlugin::ibsend;
    using CommVolumePlugin::irsend;
    using CommVolumePlugin::isend;
    using CommVolumePlugin::issend;
    using CommVolumePlugin::reduce;
    using CommVolumePlugin::rsend;
    using CommVolumePlugin::scan;
    using CommVolumePlugin::scan_single;
    using CommVolumePlugin::scatter;
    using CommVolumePlugin::scatterv;
    using CommVolumePlugin::send;
    using CommVolumePlugin::ssend;
};

using Communicator = TrackingCommunicator;

// using Communicator = kamping::Communicator<std::vector, AllReduceSinglePlugin>;

} // namespace dss_mehnert
