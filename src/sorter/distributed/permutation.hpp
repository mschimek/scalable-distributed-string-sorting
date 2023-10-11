// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <numeric>
#include <ostream>
#include <type_traits>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/exscan.hpp>
#include <tlx/die.hpp>

#include "kamping/named_parameters.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"

// todo might want to put this into a nested namespace
namespace dss_mehnert {

template <typename StringSet>
constexpr bool has_permutation_members = has_member<typename StringSet::String, StringIndex>
                                         && has_member<typename StringSet::String, PEIndex>;

template <typename StringSet>
concept PermutationStringSet = has_permutation_members<StringSet>;

template <typename String>
using AugmentedString = String::template with_members_t<StringIndex, PEIndex>;

template <typename StringSet>
using AugmentedStringSet =
    StringSet::template StringSet<AugmentedString<typename StringSet::String>>;

template <typename StringSet>
StringLcpContainer<AugmentedStringSet<StringSet>>
augment_string_container(StringLcpContainer<StringSet>&& container, size_t const rank)
    requires(!has_permutation_members<StringSet>)
{
    std::vector<AugmentedString<typename StringSet::String>> strings;
    strings.reserve(container.size());

    for (size_t index = 0; auto const& src: container.get_strings()) {
        strings.push_back(src.with_members(StringIndex{index++}, PEIndex{rank}));
    }

    return {container.release_raw_strings(), std::move(strings), container.release_lcps()};
}

class NoPermutation {};

class InputPermutation {
public:
    using size_type = std::size_t;

    using rank_type = std::size_t;
    using index_type = std::size_t;

    explicit InputPermutation() = default;

    explicit InputPermutation(std::vector<rank_type> ranks, std::vector<index_type> strings)
        : ranks_{std::move(ranks)},
          strings_{std::move(strings)} {
        assert_equal(ranks.size(), strings.size());
    }

    template <PermutationStringSet StringSet>
    explicit InputPermutation(StringSet const& ss) : ranks_(ss.size()),
                                                     strings_(ss.size()) {
        size_type i = 0;
        for (auto it = ss.begin(); it != ss.end(); ++it, ++i) {
            ranks_[i] = ss[it].PEIndex;
            strings_[i] = ss[it].stringIndex;
        }
    }

    size_type size() const { return ranks_.size(); }
    bool empty() const { return ranks_.empty(); }

    rank_type rank(size_type const n) const { return ranks_[n]; }
    index_type string(size_type const n) const { return strings_[n]; }

    std::vector<rank_type>& ranks() { return ranks_; }
    std::vector<rank_type> const& ranks() const { return ranks_; }
    std::vector<index_type>& strings() { return strings_; }
    std::vector<index_type> const& strings() const { return strings_; }

    template <typename Subcommunicators>
    void apply(
        std::span<index_type> global_permutation,
        index_type const global_index_offset,
        Subcommunicators const& comms
    ) const {
        namespace kmp = kamping;

        // todo don't really need to instantiate the permutation for this
        auto const& comm = comms.comm_root();
        std::vector<int> counts(comm.size()), offsets(comm.size());
        std::for_each(ranks_.begin(), ranks_.end(), [&](auto const rank) { ++counts[rank]; });
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);

        index_type const local_index_offset =
            comm.exscan_single(kmp::send_buf(size()), kmp::op(std::plus<>{}));
        index_type const index_offset = global_index_offset + local_index_offset;

        std::vector<std::array<index_type, 2>> send_buf(size()), recv_buf;
        for (size_type i = 0; i != size(); ++i) {
            send_buf[offsets[ranks_[i]]++] = {strings_[i], index_offset + i};
        }

        comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(counts), kmp::recv_buf(recv_buf));

        for (auto const& [local_index, global_index]: recv_buf) {
            global_permutation[local_index] = global_index;
        }
    }

private:
    std::vector<rank_type> ranks_;
    std::vector<index_type> strings_;
};

class MultiLevelPermutation {
public:
    using size_type = std::size_t;

    using rank_type = int;
    using index_type = std::size_t;

    struct RemotePermutation {
        std::vector<rank_type> ranks;
        std::vector<int> counts;
    };

    MultiLevelPermutation() = delete;

    template <PermutationStringSet StringSet>
    explicit MultiLevelPermutation(StringSet const& ss) : local_permutation_(ss.size()) {
        for (auto src = ss.begin(); auto& dst: local_permutation_) {
            dst = ss[src++].stringIndex;
        }
    }

    template <PermutationStringSet StringSet>
    void push_level(StringSet const& ss, std::vector<int> counts) {
        std::vector<int> ranks(ss.size());

        for (auto src = ss.begin(); auto& dst: ranks) {
            dst = ss[src++].PEIndex;
        }
        remote_permutations_.emplace_back(std::move(ranks), std::move(counts));
    }

    size_type depth() const { return remote_permutations_.size(); }
    void clear() { remote_permutations_.clear(); }

    std::vector<index_type> const& local_permutation() const { return local_permutation_; }
    std::vector<index_type>& local_permutation() { return local_permutation_; }

    RemotePermutation const& remote_permutation(size_type const n) const {
        return remote_permutations_[n];
    }
    RemotePermutation& remote_permutation(size_type const n) { return remote_permutations_[n]; }

    template <typename Subcommunicators>
    void apply(
        std::span<index_type> global_permutation,
        index_type const global_index_offset,
        Subcommunicators const& comms
    ) const {
        // todo maybe relax this requirement
        assert(comms.comm_root().size() > 1);

        assert_equal(
            std::distance(comms.begin(), comms.end()) + 1,
            std::ssize(remote_permutations_)
        );
        assert(comms.comm_root().is_same_on_all_ranks(global_index_offset));

        std::vector<index_type> send_buf, recv_buf;
        std::vector<int> offsets, temp_offsets;

        auto level_it = comms.rbegin();
        for (auto it = remote_permutations_.rbegin(); it != remote_permutations_.rend(); ++it) {
            auto const& [ranks, counts] = *it;

            offsets.resize(counts.size());
            std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);

            bool const is_first = it == remote_permutations_.rbegin();

            assert_equal(counts.size(), offsets.size());
            send_buf.resize(ranks.size());

            temp_offsets = offsets;
            if (is_first) {
                index_type const local_index_offset = comms.comm_root().exscan_single(
                    kamping::send_buf(ranks.size()),
                    kamping::op(std::plus<>{})
                );
                index_type const index_offset = global_index_offset + local_index_offset;

                for (size_type i = 0; i != ranks.size(); ++i) {
                    send_buf[temp_offsets[ranks[i]]++] = index_offset + i;
                }
            } else {
                assert_equal(recv_buf.size(), ranks.size());
                for (size_type i = 0; i != ranks.size(); ++i) {
                    send_buf[temp_offsets[ranks[i]]++] = recv_buf[i];
                }
            }

            auto const& comm = is_first ? comms.comm_final() : (*level_it++).comm_exchange;
            comm.alltoallv(
                kamping::send_buf(send_buf),
                kamping::send_counts(counts),
                kamping::send_displs(offsets),
                kamping::recv_buf(recv_buf)
            );
        }

        for (size_t i = 0; auto const global_index: recv_buf) {
            global_permutation[local_permutation_[i++]] = global_index;
        }
    }

private:
    std::vector<index_type> local_permutation_;
    std::vector<RemotePermutation> remote_permutations_;
};

inline std::ostream& operator<<(std::ostream& stream, InputPermutation const& permutation) {
    for (size_t i = 0; i != permutation.size(); ++i) {
        stream << "{" << permutation.rank(i) << ", " << permutation.string(i) << "}, ";
    }
    return stream;
}

inline std::ostream& operator<<(std::ostream& stream, MultiLevelPermutation const& permutation) {
    auto const& local = permutation.local_permutation();
    std::cout << "local permutation: ";
    std::copy(local.begin(), local.end(), std::ostream_iterator<size_t>(std::cout, ", "));
    std::cout << std::endl;

    for (size_t depth = 0; depth != permutation.depth(); ++depth) {
        auto const& ranks = permutation.remote_permutation(depth).ranks;
        std::cout << "remote permutation[" << depth << "]: ";
        std::copy(ranks.begin(), ranks.end(), std::ostream_iterator<int>(std::cout, ", "));
        std::cout << std::endl;
    }
    return stream;
}

} // namespace dss_mehnert
