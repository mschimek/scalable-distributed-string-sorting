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


// todo this can be merged with getStrings
template <typename StringSet, typename Iterator>
void reorder(StringSet ss, Iterator perm_begin, Iterator perm_end) {
    using std::begin;

    assert_equal(std::ssize(ss), perm_end - perm_begin);
    dss_schimek::mpi::environment env;

    std::vector<size_t> offsets(env.size());
    for (auto src = perm_begin; src != perm_end; ++src) {
        ++offsets[src->PEIndex];
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

    // using back_inserter here becuase String may not be default-insertable
    std::vector<typename StringSet::String> strings;
    strings.reserve(ss.size());
    std::transform(perm_begin, perm_end, std::back_inserter(strings), [&](auto const& x) {
        return ss[begin(ss) + offsets[x.PEIndex]++];
    });

    std::copy(strings.begin(), strings.end(), begin(ss));
}
