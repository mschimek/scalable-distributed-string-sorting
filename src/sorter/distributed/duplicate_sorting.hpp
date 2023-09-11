#pragma once

#include <algorithm>
#include <cstdint>

namespace dss_mehnert {

template <typename StringLcpPtr>
void sort_duplicates(StringLcpPtr const& strptr) {
    constexpr auto comp = [](auto const& lhs, auto const& rhs) { return lhs.index < rhs.index; };

    if (!strptr.active().empty()) {
        auto const& ss = strptr.active();

        size_t prev_length = ss.get_length(ss[ss.begin()]);
        auto curr_begin = ss.begin();

        size_t i = 1;
        for (auto it = ss.begin() + 1; it != ss.end(); ++it, ++i) {
            auto const lcp = strptr.get_lcp(i);
            auto const curr_length = ss.get_length(ss[it]);

            if (prev_length != lcp || curr_length != prev_length) {
                if (it - curr_begin > 1) {
                    std::sort(curr_begin, it, comp);
                }
                curr_begin = it;
            }
            prev_length = curr_length;
        }

        if (ss.end() - curr_begin > 1) {
            std::sort(curr_begin, ss.end(), comp);
        }
    }
}

} // namespace dss_mehnert
