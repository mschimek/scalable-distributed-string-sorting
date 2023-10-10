// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstddef>
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
            ranks_[i] = ss[it].getPEIndex();
            strings_[i] = ss[it].getStringIndex();
        }
    }

    size_type size() const { return ranks_.size(); }
    bool empty() const { return ranks_.empty(); }

    void reserve(size_type const count) {
        ranks_.reserve(count);
        strings_.reserve(count);
    }

    rank_type rank(size_type const n) const { return ranks_[n]; }
    index_type string(size_type const n) const { return strings_[n]; }

    std::vector<rank_type>& ranks() { return ranks_; }
    std::vector<rank_type> const& ranks() const { return ranks_; }
    std::vector<index_type>& strings() { return strings_; }
    std::vector<index_type> const& strings() const { return strings_; }

    void append(InputPermutation const& other) {
        ranks_.insert(ranks_.end(), other.ranks_.begin(), other.ranks_.end());
        strings_.insert(strings_.end(), other.strings_.begin(), other.strings_.end());
    }

    template <typename Communicator>
    std::vector<std::array<index_type, 2>>
    map_to_global_indices(index_type const global_rank_offset, Communicator const& comm) const {
        namespace kmp = kamping;

        std::vector<int> counts(comm.size()), offsets(comm.size());
        std::for_each(ranks_.begin(), ranks_.end(), [&](auto const rank) { ++counts[rank]; });
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);

        index_type const local_rank_offset =
            comm.exscan_single(kmp::send_buf(size()), kmp::op(std::plus<>{}));
        index_type const rank_offset = global_rank_offset + local_rank_offset;

        std::vector<std::array<index_type, 2>> send_buf(size()), recv_buf;
        for (size_type i = 0; i != size(); ++i) {
            send_buf[offsets[ranks_[i]]++] = {strings_[i], rank_offset + i};
        }

        return comm.alltoallv(kmp::send_buf(send_buf), kmp::send_counts(counts))
            .extract_recv_buffer();
    }

private:
    std::vector<rank_type> ranks_;
    std::vector<index_type> strings_;
};

inline std::ostream& operator<<(std::ostream& stream, InputPermutation const& permutation) {
    for (size_t i = 0; i != permutation.size(); ++i) {
        stream << "{" << permutation.rank(i) << ", " << permutation.string(i) << "}, ";
    }
    return stream;
}

class MultiLevelPermutation {
public:
    using size_type = std::size_t;

    using rank_type = int;
    using index_type = std::size_t;

    struct RemotePermutation {
        std::vector<rank_type> ranks;
        std::vector<int> counts;
        std::vector<int> offsets;
    };

    MultiLevelPermutation() = delete;

    template <PermutationStringSet StringSet>
    explicit MultiLevelPermutation(StringSet const& ss) : local_permutation_(ss.size()) {
        for (auto src = ss.begin(); auto& dst: local_permutation_) {
            dst = ss[src].getStringIndex();
        }
    }

    template <PermutationStringSet StringSet>
    void push_level(StringSet const& ss, std::vector<int> counts) {
        std::vector<int> ranks(ss.size()), offsets(ss.size());
        std::exclusive_scan(counts.begin(), counts.end(), offsets.begin(), 0);

        for (auto src = ss.begin(); auto& dst: ranks) {
            dst = ss[src++].getPEIndex();
        }
        remote_permutations_.emplace_back(std::move(ranks), std::move(counts), std::move(offsets));
    }

    size_type depth() const { return remote_permutations_.size(); }

    std::vector<index_type> const& local_permutation() const { return local_permutation_; }
    std::vector<index_type>& local_permutation() { return local_permutation_; }

    RemotePermutation const& remote_permutation(size_type const n) const {
        return remote_permutations_[n];
    }
    RemotePermutation& remote_permutation(size_type const n) { return remote_permutations_[n]; }

    template <typename Subcommunicators>
    std::vector<size_t> to_global_permutation(
        index_type const global_rank_offset, Subcommunicators const& comms
    ) const {
        // this function only works for full permutations, not parts thereof
        auto const& comm_root = comms.comm_root();
        if (comm_root.size() == 1) {
            return local_permutation_;
        }

        assert_equal(std::distance(comms.begin(), comms.end()) + 1, remote_permutations_.size());
        assert(comm_root.is_same_on_all_ranks(global_rank_offset));

        std::vector<index_type> send_buf, recv_buf;
        std::vector<rank_type> temp_offsets;

        auto level_it = comms.rbegin();
        for (auto it = remote_permutations_.rbegin(); it != remote_permutations_.rend(); ++it) {
            auto const& [ranks, counts, offsets] = *it;

            bool const is_first = level_it == comms.rbegin();
            bool const is_final = level_it == comms.rend();
            auto const& comm = is_final ? comm_root : (level_it++)->comm_group;

            assert_equal(ranks.size(), recv_buf.size());
            send_buf.resize(ranks.size());

            temp_offsets = offsets;
            if (is_first) {
                index_type const local_rank_offset = comm_root.exscan_single(
                    kamping::send_buf(ranks.size()),
                    kamping::op(std::plus<>{})
                );
                index_type const rank_offset = global_rank_offset + local_rank_offset;

                for (size_type i = 0; i != ranks.size(); ++i) {
                    send_buf[temp_offsets[ranks[i]]++] = rank_offset + i;
                }
            } else {
                for (size_type i = 0; i != ranks.size(); ++i) {
                    send_buf[temp_offsets[ranks[i]]++] = recv_buf[i];
                }
            }

            comm.alltoallv(
                kamping::send_buf(send_buf),
                kamping::send_counts(counts),
                kamping::send_displs(offsets),
                kamping::recv_buf(recv_buf)
            );
        }

        assert_equal(local_permutation_.size(), recv_buf.size());
        send_buf.clear(), send_buf.resize(local_permutation_.size());

        for (size_type i = 0; i != send_buf.size(); ++i) {
            send_buf[i] = local_permutation_[recv_buf[i]];
        }
        return send_buf;
    }

private:
    std::vector<index_type> local_permutation_;
    std::vector<RemotePermutation> remote_permutations_;
};

} // namespace dss_mehnert
