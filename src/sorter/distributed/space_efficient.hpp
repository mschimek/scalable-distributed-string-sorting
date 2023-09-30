// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <numeric>

#include <tlx/math/div_ceil.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"

namespace dss_mehnert {
namespace sorter {

template <
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename PartitionPolicy,
    typename BloomFilter>
class SpaceEfficientSort : private BasePrefixDoublingMergeSort<
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

    using Subcommunicators = Base::Subcommunicators;

    SpaceEfficientSort(PartitionPolicy partition, size_t const quantile_size)
        : Base{std::move(partition)},
          quantile_size_{quantile_size} {}

    template <typename StringSet>
    InputPermutation sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms)
        requires(!PermutationStringSet<StringSet>)
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
        auto const strptr = container.make_string_lcp_ptr();
        auto const& comm_root = comms.comm_root();

        this->measuring_tool_.setPhase("local_sorting");

        // todo debug total size of strings
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        if (comm_root.size() == 1) {
            this->measuring_tool_.start("write_permutation");
            InputPermutation const permutation{strptr.active()};
            this->measuring_tool_.stop("write_permutation");

            this->measuring_tool_.setPhase("none");
            return permutation;
        }

        auto const prefixes = Base::run_bloom_filter(strptr, comms, start_depth);

        this->measuring_tool_.start("compute_quantiles", "compute_quantiles");
        sample::DistPrefixes const arg{prefixes};
        auto const [quantile_sizes, quantile_offsets] =
            compute_quantiles(strptr, arg, comms.comm_root());
        this->measuring_tool_.stop("compute_quantiles", "compute_quantiles");

        InputPermutation full_permutation;
        full_permutation.reserve(strptr.size());

        for (size_t i = 0; i != quantile_sizes.size(); ++i) {
            this->measuring_tool_.setQuantile(i);
            auto const offset = quantile_offsets[i], size = quantile_sizes[i];
            auto const quantile = strptr.sub(offset, size);

            // todo use a std::span here
            std::vector<size_t> const quantile_prefixes{
                prefixes.begin() + offset,
                prefixes.begin() + offset + size};

            // todo the usage of StringContainer is slightly dodgy here
            // the strings are only actually materialized later
            StringLcpContainer<StringSet> materialized_strings{
                std::vector<typename StringSet::Char>{},
                {quantile.active().begin(), quantile.active().end()},
                {quantile.lcp(), quantile.lcp() + quantile.size()}};

            // todo would it be possible to pass `quantile` directly?
            auto sorted_quantile =
                Base::sort(std::move(materialized_strings), comms, quantile_prefixes);
            full_permutation.append(sorted_quantile.make_string_set());
        }

        this->measuring_tool_.setQuantile(0);
        return full_permutation;
    }

private:
    static constexpr size_t start_depth = 8;

    size_t quantile_size_;

    template <typename StringPtr, typename ExtraArg>
    std::pair<std::vector<size_t>, std::vector<size_t>>
    compute_quantiles(StringPtr const& strptr, ExtraArg const arg, Communicator const& comm) {
        this->measuring_tool_.start("compute_num_quantiles");
        size_t const total_size = sample::accumulate_chars(strptr.active(), arg);
        size_t const num_quantiles = comm.allreduce_single(
            kamping::send_buf({tlx::div_ceil(total_size, std::max<size_t>(1, quantile_size_))}),
            kamping::op(kamping::ops::max<>{})
        );
        this->measuring_tool_.stop("compute_num_quantiles");
        this->measuring_tool_.add(num_quantiles, "num_quantiles");

        this->measuring_tool_.start("compute_quantile_sizes");
        auto const sizes = PartitionPolicy::compute_partition(strptr, num_quantiles, arg, comm);
        std::vector<size_t> offsets(sizes.size());
        std::exclusive_scan(sizes.begin(), sizes.end(), offsets.begin(), size_t{0});
        this->measuring_tool_.stop("compute_quantile_sizes");

        return {std::move(sizes), std::move(offsets)};
    }
};

} // namespace sorter
} // namespace dss_mehnert
