// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

#include "mpi/environment.hpp"

struct StringIndexPEIndex {
    size_t stringIndex;
    size_t PEIndex;

    friend std::ostream& operator<<(std::ostream& stream, StringIndexPEIndex const& indices) {
        return stream << "[" << indices.stringIndex << ", " << indices.PEIndex << "]";
    }
};

template <typename DataType>
inline std::vector<DataType>
flatten(std::vector<std::vector<DataType>> const& dataToFlatten, const size_t totalSumElements) {
    std::vector<DataType> flattenedData;
    flattenedData.reserve(totalSumElements);
    for (auto const& curVec: dataToFlatten) {
        std::copy(curVec.begin(), curVec.end(), std::back_inserter(flattenedData));
    }
    return flattenedData;
}

template <typename DataType>
inline std::vector<DataType> flatten(std::vector<std::vector<DataType>> const& data) {
    auto op = [](auto const& sum, auto const& vec) { return sum + vec.size(); };
    return flatten(data, std::accumulate(data.begin(), data.end(), size_t{0}, op));
}

template <typename StringSet, typename Iterator>
void reorder(StringSet ss, Iterator begin, Iterator end) {
    using String = typename StringSet::String;

    dss_schimek::mpi::environment env;
    std::vector<size_t> numberInStringSet(env.size(), 0);
    std::vector<size_t> smallestIndexInPermutation(env.size(), std::numeric_limits<size_t>::max());

    for (Iterator curIt = begin; curIt != end; ++curIt) {
        auto const& indices = *curIt;
        numberInStringSet[indices.PEIndex]++;
        size_t& curSmallestIndex = smallestIndexInPermutation[indices.PEIndex];
        curSmallestIndex = std::min(curSmallestIndex, indices.stringIndex);
    }

    std::vector<size_t> startIndices(numberInStringSet.size());
    std::exclusive_scan(
        numberInStringSet.begin(),
        numberInStringSet.end(),
        startIndices.begin(),
        size_t{0}
    );

    std::vector<String> reorderedStrings;
    reorderedStrings.reserve(end - begin);
    for (Iterator curIt = begin; curIt != end; ++curIt) {
        auto const& indices = *curIt;
        const size_t stringOffset = startIndices[indices.PEIndex];
        String str =
            ss[ss.begin() + stringOffset + indices.stringIndex
               - smallestIndexInPermutation[indices.PEIndex]];
        reorderedStrings.push_back(str);
    }
    for (size_t i = 0; i < ss.size(); ++i)
        ss[ss.begin() + i] = reorderedStrings[i];
}
