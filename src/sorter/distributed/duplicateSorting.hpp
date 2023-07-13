#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "mpi/environment.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"

template <typename StringContainer>
void sortRanges(
    StringContainer& indexContainer, std::vector<std::pair<uint64_t, uint64_t>> const& ranges
) {
    using StringSet = typename StringContainer::StringSet;
    using IndexString = typename StringSet::String;
    for (auto [begin, end]: ranges) {
        std::sort(
            indexContainer.strings() + begin,
            indexContainer.strings() + end,
            [&](IndexString a, IndexString b) { return a.index < b.index; }
        );
        ;
    }
}

template <typename StringLcpPtr>
std::vector<std::pair<uint64_t, uint64_t>> getDuplicateRanges(StringLcpPtr strptr) {
    using StartEnd = std::pair<uint64_t, uint64_t>;
    // using String = typename StringLcpPtr::StringSet::String;
    std::vector<StartEnd> intervals;

    if (strptr.size() == 0)
        return intervals;

    auto ss = strptr.active();
    intervals.emplace_back(0, 0);
    uint64_t prevLength = ss.get_length(ss[ss.begin()]);
    for (size_t i = 1; i < strptr.size(); ++i) {
        const uint64_t curLcp = strptr.get_lcp(i);
        auto curString = ss[ss.begin() + i];
        const uint64_t curLength = ss.get_length(curString);
        if (curLength != curLcp || prevLength != curLcp) {
            if (intervals.back().first + 1 != i) {
                intervals.back().second = i;
                intervals.emplace_back(i, i);
            } else {
                intervals.back().first = i;
            }
        }
        prevLength = curLength;
    }
    intervals.back().second = strptr.size();
    return intervals;
}
