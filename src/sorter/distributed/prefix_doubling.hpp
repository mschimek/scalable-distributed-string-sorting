// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/bloomfilter2.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {
namespace sorter {

template <typename String, typename Permutation>
using AugmentedString = std::conditional_t<
    std::is_same_v<Permutation, SimplePermutation>,
    typename String::template with_members_t<StringIndex, PEIndex>,
    typename String::template with_members_t<CombinedIndex>>;

template <typename StringSet, typename Permutation>
using AugmentedStringSet =
    StringSet::template StringSet<AugmentedString<typename StringSet::String, Permutation>>;

template <typename Permutation, typename StringSet>
StringLcpContainer<AugmentedStringSet<StringSet, Permutation>>
augment_string_container(StringLcpContainer<StringSet>&& container, size_t const rank)
    requires(!has_permutation_members<StringSet>)
{
    std::vector<AugmentedString<typename StringSet::String, Permutation>> strings;
    strings.reserve(container.size());

    for (size_t index = 0; auto const& src: container.get_strings()) {
        if constexpr (std::is_same_v<Permutation, SimplePermutation>) {
            strings.push_back(src.with_members(StringIndex{index++}, PEIndex{rank}));
        } else {
            auto index_u32 = static_cast<uint32_t>(index++);
            strings.push_back(src.with_members(CombinedIndex{index_u32}));
        }
    }
    return {container.release_raw_strings(), std::move(strings), container.release_lcps()};
}


namespace prefix_doubling {

template <
    AlltoallStringsConfig config,
    typename RedistributionPolicy,
    typename PartitionPolicy,
    typename BloomFilter>
class BasePrefixDoublingMergeSort
    : protected BaseDistributedMergeSort<config, RedistributionPolicy, PartitionPolicy> {
public:
    using Base = BaseDistributedMergeSort<config, RedistributionPolicy, PartitionPolicy>;
    using Base::BaseDistributedMergeSort;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

    BasePrefixDoublingMergeSort(
        PartitionPolicy partition,
        RedistributionPolicy redistribution,
        bool const bloomfilter_base_case,
        bool const bloomfilter_level_dedup = false,
        mpi::OneFactorParams onefactor_params = {}
    )
        : Base{std::move(partition), std::move(redistribution), onefactor_params},
          bloomfilter_base_case_{bloomfilter_base_case},
          bloomfilter_level_dedup_{bloomfilter_level_dedup} {}

protected:
    // whether the bloom filter may try its allgather-based base case each round (see
    // bloomfilter2::BaseCaseDetector); it only fires when every PE holds at most one hash
    bool bloomfilter_base_case_ = false;

    // forward one entry per distinct hash at each intermediate grid level
    bool bloomfilter_level_dedup_ = false;

    template <typename StringPtr>
    std::vector<size_t> run_bloom_filter(
        StringPtr const& strptr, Subcommunicators const& comms, size_t const start_depth
    ) {
        // the `BloomFilter` template parameter no longer names the implementation; it only
        // selects the hasher and the communicator topology, both of which v2 takes as
        // ordinary arguments. It disappears once v1 is deleted.
        using Traits = bloomfilter2::v1_traits<BloomFilter>;

        this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
        kamping::measurements::timer().synchronize_and_start("bloomfilter_overall");
        auto detection = bloomfilter2::make_remote_duplicate_detector(
            comms,
            Traits::is_grid,
            bloomfilter_base_case_,
            bloomfilter_level_dedup_
        );
        auto const prefixes =
            bloomfilter2::compute_distinguishing_prefixes<typename Traits::hash_policy>(
                strptr,
                comms,
                start_depth,
                *detection
            );
        kamping::measurements::timer().stop_and_append();
        this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

        auto const total_prefix = std::accumulate(prefixes.begin(), prefixes.end(), size_t{0});
        this->measuring_tool_.add(total_prefix, "total_dist_prefix");

        return prefixes;
    }

    template <PermutationStringSet StringSet, typename PermutationBuilder>
    void sort(
        StringLcpContainer<StringSet>& container,
        Subcommunicators const& comms,
        std::span<size_t const> dist_prefixes,
        PermutationBuilder& builder
    ) {
        auto const& comm_root = comms.comm_root();

        if (comms.begin() == comms.end()) {
            // special case for single-level sort; consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "final_sorting");
            kamping::measurements::timer().synchronize_and_start("final_sorting");
            sample::DistPrefixes const arg{dist_prefixes};
            auto const& comm = comms.comm_final();
            auto const strptr = container.make_string_lcp_ptr();
            auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, comm);
            Base::exchange_and_merge(container, send_counts, arg, builder, comm);
            kamping::measurements::timer().stop_and_append();
            this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
            this->measuring_tool_.setRound(0);
            return;
        }

        if constexpr (!Subcommunicators::is_single_level) {
            size_t round = 0;
            auto level_it = comms.begin();
            {
                // first level of multi-level sort; consider distinguishing prefixes
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                kamping::measurements::timer().synchronize_and_start("partial_sorting");
                sample::DistPrefixes const arg{dist_prefixes};
                auto const level = *level_it++;
                auto const& comm = level.comm_exchange;
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, level);
                Base::exchange_and_merge(container, send_counts, arg, builder, comm);
                kamping::measurements::timer().stop_and_append();
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                this->measuring_tool_.setRound(++round);
            }

            for (; level_it != comms.end(); ++level_it) {
                // intermediate level of multi-level sort; don't consider distinguishing prefixes
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                kamping::measurements::timer().synchronize_and_start("partial_sorting");
                sample::NoExtraArg const arg;
                auto const level = *level_it;
                auto const& comm = level.comm_exchange;
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, level);
                Base::exchange_and_merge(container, send_counts, arg, builder, comm);
                kamping::measurements::timer().stop_and_append();
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                this->measuring_tool_.setRound(++round);
            }

            {
                this->measuring_tool_.start("sort_globally", "final_sorting");
                kamping::measurements::timer().synchronize_and_start("final_sorting");
                // final level of multi-level sort; don't consider distinguishing prefixes
                sample::NoExtraArg const arg;
                auto const& comm = comms.comm_final();
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, comm);
                Base::exchange_and_merge(container, send_counts, arg, builder, comm);
                kamping::measurements::timer().stop_and_append();
                this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
                this->measuring_tool_.setRound(0);
            }
        } else {
            tlx_die("unreachable for a purely single level implementation");
        }
    }

    template <PermutationStringSet StringSet, typename PermutationBuilder>
    typename PermutationBuilder::Permutation
    write_permutation(StringSet const& ss, PermutationBuilder& builder) {
        this->measuring_tool_.start("write_permutation");
        auto const permutation = builder.build(ss);
        this->measuring_tool_.stop("write_permutation");

        this->measuring_tool_.setPhase("none");
        return permutation;
    }
};

template <typename Permutation>
class PermutationBuilder;

template <>
class PermutationBuilder<SimplePermutation> {
public:
    using Permutation = SimplePermutation;

    template <PermutationStringSet StringSet>
    explicit PermutationBuilder(StringSet const& ss) {}

    template <PermutationStringSet StringSet>
    void push(StringSet const& ss, std::vector<size_t> const&) {}

    template <PermutationStringSet StringSet>
    Permutation build(StringSet const& ss) {
        return Permutation{ss};
    }
};

template <>
class PermutationBuilder<MultiLevelPermutation> {
public:
    using Permutation = MultiLevelPermutation;

    template <PermutationStringSet StringSet>
    explicit PermutationBuilder(StringSet const& ss) : local_{ss} {}

    template <PermutationStringSet StringSet>
    void push(StringSet const& ss, std::vector<size_t> counts) {
        remote_.emplace_back(ss, std::vector<int>{counts.begin(), counts.end()});
    }

    template <PermutationStringSet StringSet>
    Permutation build(StringSet const& ss) {
        return {std::move(this->local_), std::move(this->remote_)};
    }

protected:
    Permutation::LocalPermutation local_;
    std::vector<Permutation::RemotePermutation> remote_;
};

template <
    AlltoallStringsConfig config,
    typename RedistributionPolicy,
    typename PartitionPolicy,
    typename BloomFilter,
    typename Permutation>
class PrefixDoublingMergeSort : private BasePrefixDoublingMergeSort<
                                    config,
                                    RedistributionPolicy,
                                    PartitionPolicy,
                                    BloomFilter> {
public:
    using Base =
        BasePrefixDoublingMergeSort<config, RedistributionPolicy, PartitionPolicy, BloomFilter>;

    using Base::Base;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

    template <typename StringSet>
    Permutation sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms)
        requires(!has_permutation_members<StringSet>)
    {
        this->measuring_tool_.start("augment_container");
        auto const rank = comms.comm_root().rank();
        auto augmented_container =
            augment_string_container<Permutation>(std::move(container), rank);
        this->measuring_tool_.stop("augment_container");

        return sort(std::move(augmented_container), comms);
    }

    template <PermutationStringSet StringSet>
    Permutation sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms) {
        this->measuring_tool_.setPhase("local_sorting");

        auto const strptr = container.make_string_lcp_ptr();
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        kamping::measurements::timer().synchronize_and_start("sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
        kamping::measurements::timer().stop_and_append();
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        PermutationBuilder<Permutation> builder{strptr.active()};

        if (comms.comm_root().size() != 1) {
            auto const prefixes = Base::run_bloom_filter(strptr, comms, start_depth);
            Base::sort(container, comms, prefixes, builder);
            this->measuring_tool_.setRound(0);
        }

        return Base::write_permutation(container.make_string_set(), builder);
    }

private:
    static constexpr size_t start_depth = 8;
};

namespace _internal {

template <typename StringSet>
StringLcpContainer<StringSet> receive_strings(
    StringSet const& ss, SimplePermutation const& permutation, Communicator const& comm
) {
    std::vector<int> req_sizes(comm.size());
    for (auto const rank: permutation.ranks()) {
        ++req_sizes[rank];
    }

    std::vector<size_t> offsets(comm.size());
    std::exclusive_scan(req_sizes.begin(), req_sizes.end(), offsets.begin(), size_t{0});

    std::vector<size_t> requests(permutation.size());
    for (size_t i = 0; i != permutation.size(); ++i) {
        requests[offsets[permutation.rank(i)]++] = permutation.string(i);
    }

    auto result = comm.alltoallv(
        kamping::send_buf(requests),
        kamping::send_counts(req_sizes),
        kamping::recv_counts_out()
    );
    auto recv_requests = result.extract_recv_buffer();
    auto recv_req_sizes = result.extract_recv_counts();

    std::vector<unsigned char> raw_strs;
    std::vector<size_t> raw_str_sizes(comm.size());
    for (size_t rank = 0, offset = 0; rank != comm.size(); ++rank) {
        for (size_t i = 0; i < static_cast<size_t>(recv_req_sizes[rank]); ++i, ++offset) {
            auto const& str = ss.at(recv_requests[offset]);
            auto str_len = ss.get_length(str);
            auto str_chars = ss.get_chars(str, 0);
            raw_strs.insert(raw_strs.end(), str_chars, str_chars + str_len);
            raw_strs.push_back(0);
            raw_str_sizes[rank] += str_len + 1;
        }
    }

    constexpr auto alltoall_kind = mpi::AlltoallvCombinedKind::native;
    auto recv_buf_char = comm.template alltoallv_combined<alltoall_kind>(raw_strs, raw_str_sizes);
    return StringLcpContainer<StringSet>{std::move(recv_buf_char)};
}

template <typename StringSet>
std::vector<typename StringSet::String> compute_reordered_strings(
    StringSet const& ss, SimplePermutation const& permutation, Communicator const& comm
) {
    std::vector<size_t> offsets(comm.size());
    for (auto const rank: permutation.ranks()) {
        ++offsets[rank];
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

    std::vector<typename StringSet::String> strings;
    strings.reserve(ss.size());
    for (auto const rank: permutation.ranks()) {
        strings.push_back(ss.at(offsets[rank]++));
    }
    return strings;
}

} // namespace _internal


// todo could also use the gridwise subcommunicators here
// Apply a permutation generated by prefix doubling merge-sort.
template <typename StringSet>
StringLcpContainer<StringSet> apply_permutation(
    StringSet const& ss, SimplePermutation const& permutation, Communicator const& comm
) {
    auto output_container = _internal::receive_strings(ss, permutation, comm);

    auto const oss = output_container.make_string_set();
    output_container.set(_internal::compute_reordered_strings(oss, permutation, comm));
    output_container.make_contiguous();

    assert_equal(output_container.size(), permutation.size());
    return output_container;
}

} // namespace prefix_doubling
} // namespace sorter
} // namespace dss_mehnert
