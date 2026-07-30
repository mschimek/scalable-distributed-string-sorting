// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Statistics reported about the input itself, measured across PE boundaries: the total LCP and
// the total distinguishing prefix of the global string sequence.

#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <numeric>
#include <vector>

#include "dss/mpi/communicator.hpp"
#include "dss/mpi/is_sorted.hpp"
#include "dss/util/measuringTool.hpp"

namespace dss_mehnert {
namespace bench {

template <typename Container>
void count_prefix_lengths(Container& container, Communicator const& comm) {
    using namespace kamping;

    using Char = Container::Char;
    std::vector<Char> const last_string =
        container.empty() ? std::vector<Char>{0} : container.get_raw_string(container.size() - 1);
    auto const pred_string = get_predecessor(last_string, container.empty(), comm);

    size_t first_lcp = 0;
    if (pred_string && !container.empty()) {
        auto const first_string = container.get_raw_string(0);
        first_lcp = dss_schimek::calc_lcp(pred_string->data(), first_string.data());
    }

    size_t const last_lcp = container.empty() ? 0 : container.lcps().back();
    auto const pred_lcp =
        container.size() == 1 ? first_lcp : get_predecessor(last_lcp, container.empty(), comm);

    auto const begin = container.lcps().begin(), end = container.lcps().end();
    auto const local_lcp = first_lcp + std::accumulate(begin, end, size_t{0});

    auto const dist = [](auto const lcp1, auto const lcp2) { return std::max(lcp1, lcp2) + 1; };

    auto local_dist = container.empty() || !pred_lcp ? 0 : dist(*pred_lcp, first_lcp);
    if (!container.empty()) {
        local_dist =
            std::transform_reduce(std::next(begin), end, begin, size_t{0}, std::plus<>{}, dist);
    }

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    measuring_tool.add(local_lcp, "global_lcp_sum");
    measuring_tool.add(local_dist, "global_dist_prefix");
}

} // namespace bench
} // namespace dss_mehnert
