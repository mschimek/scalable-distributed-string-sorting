// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

#include "merge/bingmann-lcp_losertree.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {

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

template <bool is_compressed, typename StringSet>
struct MergeResult {
    StringLcpContainer<StringSet> container;
};

template <typename StringSet>
struct MergeResult<true, StringSet> {
    StringLcpContainer<StringSet> container;
    std::vector<size_t> saved_lcps;
};

template <bool is_compressed, size_t K, typename StringLcpPtr>
static inline MergeResult<is_compressed, typename StringLcpPtr::StringSet> multiway_merge(
    StringLcpPtr const& input_string_ptr,
    std::vector<size_t>& interval_offsets,
    std::vector<size_t>& interval_sizes
) {
    assert(K == pow2roundup(interval_sizes.size()));
    if (input_string_ptr.size() == 0) {
        return {};
    }

    interval_offsets.resize(K, 0);
    interval_sizes.resize(K, 0);

    StringLcpContainer<typename StringLcpPtr::StringSet> sorted_strings;
    sorted_strings.resize_strings(input_string_ptr.size());

    using StringSet = typename StringLcpPtr::StringSet;
    using MergeAdapter = dss_schimek::StringLcpPtrMergeAdapter<StringSet>;
    using LoserTree = dss_schimek::LcpStringLoserTree_<K, StringSet>;

    MergeAdapter in_ptr{input_string_ptr.active(), input_string_ptr.lcp()};
    MergeAdapter out_ptr{sorted_strings.make_string_set(), sorted_strings.lcp_array()};
    LoserTree loser_tree{in_ptr, interval_offsets, interval_sizes};

    if constexpr (is_compressed) {
        std::vector<size_t> old_lcps;
        loser_tree.writeElementsToStream(out_ptr, input_string_ptr.size(), old_lcps);
        return {std::move(sorted_strings), std::move(old_lcps)};
    } else {
        loser_tree.writeElementsToStream(out_ptr, input_string_ptr.size());
        return {std::move(sorted_strings)};
    }
}

template <bool is_compressed, typename StringLcpPtr>
static inline MergeResult<is_compressed, typename StringLcpPtr::StringSet> choose_merge(
    StringLcpPtr const& recv_string_ptr,
    std::vector<size_t>& interval_offsets,
    std::vector<size_t>& interval_sizes
) {
    assert(interval_sizes.size() == interval_offsets.size());
    auto merge_k = [&]<size_t K>() {
        return multiway_merge<is_compressed, K>(recv_string_ptr, interval_offsets, interval_sizes);
    };
    switch (pow2roundup(interval_sizes.size())) {
        case 1:
            return merge_k.template operator()<1>();
        case 2:
            return merge_k.template operator()<2>();
        case 4:
            return merge_k.template operator()<4>();
        case 8:
            return merge_k.template operator()<8>();
        case 16:
            return merge_k.template operator()<16>();
        case 32:
            return merge_k.template operator()<32>();
        case 64:
            return merge_k.template operator()<64>();
        case 128:
            return merge_k.template operator()<128>();
        case 256:
            return merge_k.template operator()<256>();
        case 512:
            return merge_k.template operator()<512>();
        case 1024:
            return merge_k.template operator()<1024>();
        case 2048:
            return merge_k.template operator()<2048>();
        case 4096:
            return merge_k.template operator()<4096>();
        case 8192:
            return merge_k.template operator()<8192>();
        case 16384:
            return merge_k.template operator()<16384>();
        default:
            // todo consider increasing this to 2^15
            std::cout << "Error in merge: K is not 2^i for i in {0,...,14} " << std::endl;
            std::abort();
    }
}

} // namespace dss_mehnert
