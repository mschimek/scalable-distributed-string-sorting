// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include <tlx/die.hpp>

#include "merge/bingmann-lcp_losertree.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {
namespace merge {

inline size_t pow2roundup(size_t x) {
    if (x == 0u)
        return 1u;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

template <bool is_compressed>
struct MergeResult {};

template <>
struct MergeResult<true> {
    std::vector<size_t> saved_lcps;
};

template <bool is_compressed, size_t K, typename StringSet>
inline MergeResult<is_compressed>
multiway_merge(StringLcpContainer<StringSet>& input_strings, std::vector<size_t>& interval_sizes) {
    assert_equal(K, pow2roundup(interval_sizes.size()));
    if (input_strings.empty()) {
        return {};
    }

    interval_sizes.resize(K, 0);

    using MergeAdapter = dss_schimek::StringLcpPtrMergeAdapter<StringSet>;
    using LoserTree = dss_schimek::LcpStringLoserTree_<K, StringSet>;

    std::vector<typename StringSet::String> sorted_strings(input_strings.size());
    StringSet sorted_string_set{
        sorted_strings.data(),
        sorted_strings.data() + sorted_strings.size()};
    std::vector<size_t> sorted_lcps(input_strings.size());

    MergeAdapter in_ptr{input_strings.make_string_set(), input_strings.lcp_array()};
    MergeAdapter out_ptr{sorted_string_set, sorted_lcps.data()};
    LoserTree loser_tree{in_ptr, interval_sizes};

    if constexpr (is_compressed) {
        std::vector<size_t> saved_lcps;
        loser_tree.writeElementsToStream(out_ptr, input_strings.size(), saved_lcps);
        input_strings.set(std::move(sorted_strings));
        input_strings.set(std::move(sorted_lcps));
        return {saved_lcps};
    } else {
        loser_tree.writeElementsToStream(out_ptr, input_strings.size());
        input_strings.set(std::move(sorted_strings));
        input_strings.set(std::move(sorted_lcps));
        return {};
    }
}

template <bool is_compressed, typename StringSet>
static inline MergeResult<is_compressed>
choose_merge(StringLcpContainer<StringSet>& recv_strings, std::vector<size_t>& interval_sizes) {
    auto merge_k = [&]<size_t K>() {
        return multiway_merge<is_compressed, K, StringSet>(recv_strings, interval_sizes);
    };

    switch (pow2roundup(interval_sizes.size())) {
        // clang-format off
        case 1:     return merge_k.template operator()<1>();
        case 2:     return merge_k.template operator()<2>();
        case 4:     return merge_k.template operator()<4>();
        case 8:     return merge_k.template operator()<8>();
        case 16:    return merge_k.template operator()<16>();
        case 32:    return merge_k.template operator()<32>();
        case 64:    return merge_k.template operator()<64>();
        case 128:   return merge_k.template operator()<128>();
        case 256:   return merge_k.template operator()<256>();
        case 512:   return merge_k.template operator()<512>();
        case 1024:  return merge_k.template operator()<1024>();
        case 2048:  return merge_k.template operator()<2048>();
        case 4096:  return merge_k.template operator()<4096>();
        case 8192:  return merge_k.template operator()<8192>();
        case 16384: return merge_k.template operator()<16384>();
        case 32768: return merge_k.template operator()<32768>();
        case 65536: return merge_k.template operator()<65536>();
        // clang-format on
        default:
            // todo consider increasing this to 2^15
            tlx_die("Error in merge: K is not 2^i for i in {0,...,14} ");
    }
}

} // namespace merge
} // namespace dss_mehnert
