#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace dss_mehnert {

template <typename StringLcpPtr>
void sort_duplicates(StringLcpPtr const& strptr) {
    constexpr auto comp = [](auto const& lhs, auto const& rhs) { return lhs.index < rhs.index; };

    auto const& ss = strptr.active();
    size_t i = 0, begin = 0, prev_length = 0;
    for (auto it = ss.begin(); it != ss.end(); ++it, ++i) {
        auto const& string = ss[it];
        auto const lcp = strptr.get_lcp(i);
        auto const curr_length = ss.get_length(string);

        if (prev_length != lcp || curr_length != lcp) {
            if (begin + 1 != i) {
                std::sort(ss.begin() + begin, ss.begin() + i, comp);
                begin = i;
            }
        }

        prev_length = curr_length;
    }

    if (begin + 2 < ss.size()) {
        std::sort(ss.begin() + begin, ss.end(), comp);
    }
}

} // namespace dss_mehnert
