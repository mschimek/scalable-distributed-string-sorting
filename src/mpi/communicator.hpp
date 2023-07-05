// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>

#include <kamping/assertion_levels.hpp>
#include <kamping/communicator.hpp>
#include <kamping/named_parameter_check.hpp>
#include <kamping/named_parameter_types.hpp>
#include <kamping/plugin_helpers.hpp>

namespace dss_mehnert {

/// @brief A plugin providing `allreduce_single`, a simple wrapper around `allreduce`.
template <typename Comm>
class AllReducePlugin : public kamping::plugins::PluginBase<Comm, AllReducePlugin> {
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

// todo implement lcp compression as plugin
using Communicator = kamping::Communicator<std::vector, AllReducePlugin>;

} // namespace dss_mehnert
