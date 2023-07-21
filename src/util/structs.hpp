// (c) 2019 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <vector>

#include "mpi/synchron.hpp"

struct StringIndexPEIndex {
    size_t stringIndex;
    size_t PEIndex;
    StringIndexPEIndex(const size_t stringIndex, const size_t PEIndex)
        : stringIndex(stringIndex),
          PEIndex(PEIndex) {}
};

std::ostream& operator<<(std::ostream& stream, StringIndexPEIndex const& indices) {
    return stream << "[" << indices.stringIndex << ", " << indices.PEIndex << "]";
}

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
inline std::vector<DataType> flatten(std::vector<std::vector<DataType>> const& dataToFlatten) {
    size_t totalSumElements = 0;
    for (auto const& curVec: dataToFlatten) totalSumElements += curVec.size();

    return flatten(dataToFlatten, totalSumElements);
}

template <typename StringSet, typename Iterator>
void reorder(StringSet ss, Iterator begin, Iterator end) {
    using String = typename StringSet::String;

    dss_schimek::mpi::environment env;
    std::vector<String> reorderedStrings;
    std::vector<size_t> numberInStringSet(env.size(), 0);
    std::vector<size_t> startIndexInStringSet;
    std::vector<size_t> smallestIndexInPermutation(env.size(), std::numeric_limits<size_t>::max());

    reorderedStrings.reserve(end - begin);
    for (Iterator curIt = begin; curIt != end; ++curIt) {
        auto const& indices = *curIt;
        numberInStringSet[indices.PEIndex]++;
        size_t& curSmallestIndex = smallestIndexInPermutation[indices.PEIndex];
        curSmallestIndex = std::min(curSmallestIndex, indices.stringIndex);
    }
    startIndexInStringSet.push_back(0);
    std::partial_sum(
        numberInStringSet.begin(),
        numberInStringSet.end() - 1,
        std::back_inserter(startIndexInStringSet)
    );

    for (Iterator curIt = begin; curIt != end; ++curIt) {
        auto const& indices = *curIt;
        const size_t stringOffset = startIndexInStringSet[indices.PEIndex];
        String str =
            ss[ss.begin() + stringOffset + indices.stringIndex
               - smallestIndexInPermutation[indices.PEIndex]];
        reorderedStrings.push_back(str);
    }
    for (size_t i = 0; i < ss.size(); ++i) ss[ss.begin() + i] = reorderedStrings[i];
}
