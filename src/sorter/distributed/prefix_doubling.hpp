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
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {
namespace sorter {

template <
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename PartitionPolicy,
    typename BloomFilter>
class BasePrefixDoublingMergeSort : protected BaseDistributedMergeSort<
                                        RedistributionPolicy,
                                        AllToAllStringPolicy,
                                        PartitionPolicy> {
public:
    using Base =
        BaseDistributedMergeSort<RedistributionPolicy, AllToAllStringPolicy, PartitionPolicy>;

    using Base::Base;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

protected:
    template <typename StringPtr>
    std::vector<size_t> run_bloom_filter(
        StringPtr const& strptr, Subcommunicators const& comms, size_t const start_depth
    ) {
        this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
        BloomFilter filter{comms, strptr.size()};
        auto const prefixes = filter.compute_distinguishing_prefixes(strptr, comms, start_depth);
        this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

        return prefixes;
    }

    template <PermutationStringSet StringSet>
    void sort(
        StringLcpContainer<StringSet>& container,
        Subcommunicators const& comms,
        std::span<size_t const> dist_prefixes
    ) {
        auto const& comm_root = comms.comm_root();

        if (comms.begin() == comms.end()) {
            // special case for single-level sort; consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "final_sorting");
            sample::DistPrefixes const arg{dist_prefixes};
            auto const& comm = comms.comm_final();
            auto const strptr = container.make_string_lcp_ptr();
            auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, comm);
            Base::exchange_and_merge(container, send_counts, arg, comm);
            this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
            return;
        }

        if constexpr (!Subcommunicators::is_single_level) {
            size_t round = 0;
            auto level_it = comms.begin();
            {
                // first level of multi-level sort; consider distinguishing prefixes
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                sample::DistPrefixes const arg{dist_prefixes};
                auto const level = *level_it++;
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, level);
                Base::exchange_and_merge(container, send_counts, arg, level.comm_exchange);
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                this->measuring_tool_.setRound(++round);
            }

            for (; level_it != comms.end(); ++level_it) {
                // intermediate level of multi-level sort; don't consider distinguishing prefixes
                this->measuring_tool_.start("sort_globally", "partial_sorting");
                sample::NoExtraArg const arg;
                auto const level = *level_it;
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, level);
                Base::exchange_and_merge(container, send_counts, arg, level.comm_exchange);
                this->measuring_tool_.stop("sort_globally", "partial_sorting", comm_root);
                this->measuring_tool_.setRound(++round);
            }

            {
                this->measuring_tool_.start("sort_globally", "final_sorting");
                // final level of multi-level sort; don't consider distinguishing prefixes
                sample::NoExtraArg const arg;
                auto const& comm = comms.comm_final();
                auto const strptr = container.make_string_lcp_ptr();
                auto const send_counts = Base::compute_sorted_send_counts(strptr, arg, comm);
                Base::exchange_and_merge(container, send_counts, arg, comm);
                this->measuring_tool_.stop("sort_globally", "final_sorting", comm_root);
            }
        } else {
            tlx_die("unreachable for for a purely single level implementation");
        }
    }

    template <PermutationStringSet StringSet>
    InputPermutation write_permutation(StringSet const& ss) {
        this->measuring_tool_.start("write_permutation");
        InputPermutation const permutation{ss};
        this->measuring_tool_.stop("write_permutation");

        this->measuring_tool_.setPhase("none");
        return permutation;
    }
};

template <
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename PartitionPolicy,
    typename BloomFilter>
class PrefixDoublingMergeSort : private BasePrefixDoublingMergeSort<
                                    RedistributionPolicy,
                                    AllToAllStringPolicy,
                                    PartitionPolicy,
                                    BloomFilter> {
public:
    using Base = BasePrefixDoublingMergeSort<
        RedistributionPolicy,
        AllToAllStringPolicy,
        PartitionPolicy,
        BloomFilter>;

    using Base::Base;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;

    template <typename StringSet>
    InputPermutation sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms)
        requires(!has_permutation_members<StringSet>)
    {
        this->measuring_tool_.start("augment_container");
        auto const rank = comms.comm_root().rank();
        auto augmented_container = augment_string_container(std::move(container), rank);
        this->measuring_tool_.stop("augment_container");

        return sort(std::move(augmented_container), comms);
    }

    template <PermutationStringSet StringSet>
    InputPermutation
    sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms) {
        this->measuring_tool_.setPhase("local_sorting");

        auto const strptr = container.make_string_lcp_ptr();
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        if (comms.comm_root().size() == 1) {
            return Base::write_permutation(strptr.active());
        } else {
            auto const prefixes = Base::run_bloom_filter(strptr, comms, start_depth);
            Base::sort(container, comms, prefixes);

            this->measuring_tool_.setRound(0);
            return Base::write_permutation(container.make_string_set());
        }
    }

private:
    static constexpr size_t start_depth = 8;
};

namespace _internal {

// todo maybe move all this stuff to permutation.hpp
template <typename StringSet>
StringLcpContainer<StringSet> receive_strings(
    StringLcpContainer<StringSet>&& container,
    InputPermutation const& permutation,
    Communicator const& comm
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

    auto result = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(req_sizes));
    auto recv_requests = result.extract_recv_buffer();
    auto recv_req_sizes = result.extract_recv_counts();

    auto const& ss = container.make_string_set();
    std::vector<unsigned char> raw_strs;
    std::vector<size_t> raw_str_sizes(comm.size());
    for (size_t rank = 0, offset = 0; rank < comm.size(); ++rank) {
        for (size_t i = 0; i < static_cast<size_t>(recv_req_sizes[rank]); ++i, ++offset) {
            auto const& str = ss[ss.begin() + recv_requests[offset]];
            auto str_len = ss.get_length(str) + 1;
            auto str_chars = ss.get_chars(str, 0);
            std::copy_n(str_chars, str_len, std::back_inserter(raw_strs));
            raw_str_sizes[rank] += str_len;
        }
    }

    using MPIAlltoall = dss_schimek::mpi::AllToAllvSmall;
    using MPIRoutine = dss_schimek::mpi::AllToAllvCombined<MPIAlltoall>;
    auto recv_chars = MPIRoutine::alltoallv(raw_strs.data(), raw_str_sizes, comm);

    return StringLcpContainer<StringSet>{std::move(recv_chars)};
}

template <typename StringSet>
std::vector<typename StringSet::String> compute_reordered_strings(
    StringSet const& ss, InputPermutation const& permutation, Communicator const& comm
) {
    std::vector<size_t> offsets(comm.size());
    for (auto const rank: permutation.ranks()) {
        ++offsets[rank];
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

    std::vector<typename StringSet::String> strings;
    strings.reserve(ss.size());
    std::transform(
        permutation.ranks().begin(),
        permutation.ranks().end(),
        std::back_inserter(strings),
        [&](auto const rank) { return ss.at(offsets[rank]++); }
    );

    return strings;
}

} // namespace _internal


// todo could also use the gridwise subcommunicators here
// Apply a permutation generated by prefix doubling merge-sort.
template <typename StringSet>
StringLcpContainer<StringSet> apply_permutation(
    StringLcpContainer<StringSet>&& container,
    InputPermutation const& permutation,
    Communicator const& comm
) {
    auto output_container = _internal::receive_strings(std::move(container), permutation, comm);
    assert_equal(output_container.size(), permutation.size());

    output_container.set(
        _internal::compute_reordered_strings(output_container.make_string_set(), permutation, comm)
    );

    return output_container;
}

} // namespace sorter
} // namespace dss_mehnert
