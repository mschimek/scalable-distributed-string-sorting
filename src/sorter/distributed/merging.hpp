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
    if (x == 0u) return 1u;
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    x |= x >> 32;
    return x + 1;
}

template <typename AllToAllStringPolicy, size_t K, typename StringSet>
static inline dss_schimek::StringLcpContainer<StringSet> merge(
    dss_schimek::StringLcpContainer<StringSet>&& recv_string_cont,
    std::vector<std::pair<size_t, size_t>>& ranges,
    const size_t num_recv_elems
) {
    if (recv_string_cont.empty()) return dss_schimek::StringLcpContainer<StringSet>();

    // pad with empty ranges up to next power of two
    size_t num_missing = pow2roundup(ranges.size()) - ranges.size();
    std::fill_n(std::back_inserter(ranges), num_missing, std::make_pair(0, 0));

    std::vector<typename StringSet::String> sorted_string(recv_string_cont.size());
    std::vector<size_t> sorted_lcp(recv_string_cont.size());
    StringSet ss = recv_string_cont.make_string_set();

    dss_schimek::StringLcpPtrMergeAdapter<StringSet> mergeAdapter(ss, recv_string_cont.lcp_array());
    dss_schimek::LcpStringLoserTree_<K, StringSet> loser_tree(mergeAdapter, ranges.data());
    StringSet sortedSet(sorted_string.data(), sorted_string.data() + sorted_string.size());
    dss_schimek::StringLcpPtrMergeAdapter out_(sortedSet, sorted_lcp.data());

    std::vector<size_t> oldLcps;
    if (AllToAllStringPolicy::PrefixCompression) {
        loser_tree.writeElementsToStream(out_, num_recv_elems, oldLcps);
    } else {
        loser_tree.writeElementsToStream(out_, num_recv_elems);
    }

    dss_schimek::StringLcpContainer<StringSet> sorted_string_cont;
    //(std::move(recv_string_cont));
    sorted_string_cont.set(std::move(recv_string_cont.raw_strings()));
    sorted_string_cont.set(std::move(sorted_string));
    sorted_string_cont.set(std::move(sorted_lcp));
    sorted_string_cont.setSavedLcps(std::move(oldLcps));

    return sorted_string_cont;
}

template <typename AllToAllStringPolicy, typename StringLcpContainer>
static inline StringLcpContainer choose_merge(
    StringLcpContainer&& recv_string_cont,
    std::vector<std::pair<size_t, size_t>>& ranges,
    size_t num_recv_elems
) {
    auto merge_k = [=, &ranges]<size_t K>(StringLcpContainer&& container) {
        return merge<AllToAllStringPolicy, K>(
            std::forward<StringLcpContainer>(container),
            ranges,
            num_recv_elems
        );
    };
    const size_t next_pow_2 = pow2roundup(ranges.size());
    switch (next_pow_2) {
        case 1:
            return merge_k.template operator()<1>(std::move(recv_string_cont));
        case 2:
            return merge_k.template operator()<2>(std::move(recv_string_cont));
        case 4:
            return merge_k.template operator()<4>(std::move(recv_string_cont));
        case 8:
            return merge_k.template operator()<8>(std::move(recv_string_cont));
        case 16:
            return merge_k.template operator()<16>(std::move(recv_string_cont));
        case 32:
            return merge_k.template operator()<32>(std::move(recv_string_cont));
        case 64:
            return merge_k.template operator()<64>(std::move(recv_string_cont));
        case 128:
            return merge_k.template operator()<128>(std::move(recv_string_cont));
        case 256:
            return merge_k.template operator()<256>(std::move(recv_string_cont));
        case 512:
            return merge_k.template operator()<512>(std::move(recv_string_cont));
        case 1024:
            return merge_k.template operator()<1024>(std::move(recv_string_cont));
        case 2048:
            return merge_k.template operator()<2048>(std::move(recv_string_cont));
        case 4096:
            return merge_k.template operator()<4096>(std::move(recv_string_cont));
        case 8192:
            return merge_k.template operator()<8192>(std::move(recv_string_cont));
        case 16384:
            return merge_k.template operator()<16384>(std::move(recv_string_cont));
        default:
            // todo consider increasing this to 2^15
            std::cout << "Error in merge: K is not 2^i for i in {0,...,14} " << std::endl;
            std::abort();
    }
}

} // namespace dss_mehnert
