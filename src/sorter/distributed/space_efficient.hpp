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

// this struct is required to disambiguate the PartitionPolicy used
// in sorting and the one use to compute quantiles
template <typename PartitionPolicy>
struct QuantilePolicyBase : public PartitionPolicy {};

template <
    AlltoallStringsConfig config,
    typename RedistributionPolicy,
    typename PartitionPolicy,
    typename QuantilePolicy,
    typename BloomFilter>
class SpaceEfficientSort : private QuantilePolicyBase<QuantilePolicy>,
                           private BasePrefixDoublingMergeSort<
                               config,
                               RedistributionPolicy,
                               PartitionPolicy,
                               BloomFilter> {
public:
    using QuantilePolicy_ = QuantilePolicyBase<QuantilePolicy>;
    using Base =
        BasePrefixDoublingMergeSort<config, RedistributionPolicy, PartitionPolicy, BloomFilter>;

    using Subcommunicators = Base::Subcommunicators;

    SpaceEfficientSort(
        PartitionPolicy partition, QuantilePolicy quantile_partition, size_t const quantile_size
    )
        : QuantilePolicy_{std::move(quantile_partition)},
          Base{std::move(partition)},
          quantile_size_{quantile_size} {}

    template <typename StringSet>
    std::vector<size_t>
    sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms)
        requires(!PermutationStringSet<StringSet>)
    {
        this->measuring_tool_.start("augment_container");
        auto const rank = comms.comm_root().rank();
        auto augmented_container = augment_string_container(std::move(container), rank);
        this->measuring_tool_.stop("augment_container");

        return sort(std::move(augmented_container), comms);
    }

    template <PermutationStringSet StringSet>
    std::vector<size_t>
    sort(StringLcpContainer<StringSet>&& container, Subcommunicators const& comms) {
        namespace kmp = kamping;

        auto const strptr = container.make_string_lcp_ptr();
        auto const& comm_root = comms.comm_root();

        this->measuring_tool_.setPhase("local_sorting");
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI2(strptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comm_root);

        std::vector<size_t> global_permutation(strptr.size());

        if (comm_root.size() == 1) {
            this->measuring_tool_.start("sort_globally", "write_global_permutation");
            for (size_t i = 0; auto const& str: strptr.active()) {
                global_permutation[str.stringIndex] = i++;
            }
            this->measuring_tool_.stop("sort_globally", "write_global_permutation");

            this->measuring_tool_.setPhase("none");
            return global_permutation;
        }

        auto const prefixes = Base::run_bloom_filter(strptr, comms, start_depth);

        this->measuring_tool_.start("compute_quantiles", "compute_quantiles");
        sample::DistPrefixes const arg{prefixes};
        auto const quantile_sizes = compute_quantiles(strptr, arg, comm_root);
        this->measuring_tool_.stop("compute_quantiles", "compute_quantiles", comm_root);

        this->measuring_tool_.start("sort_quantiles", "sort_quantiles_overall");

        // todo allow this to be selected externally through a template parameter
        // _internal::SimplePermutationBuilder<InputPermutation> builder{strptr.active()};
        _internal::SimplePermutationBuilder<MultiLevelPermutation> builder{strptr.active()};

        for (size_t i = 0, local_offset = 0, global_offset = 0; i != quantile_sizes.size(); ++i) {
            this->measuring_tool_.setQuantile(i);
            this->measuring_tool_.start("sort_globally", "quantile_overall");
            this->measuring_tool_.start("sort_globally", "sort_quantile");
            auto const size = quantile_sizes[i];
            auto const quantile = strptr.sub(local_offset, size);
            std::span const quantile_prefixes{prefixes.begin() + local_offset, size};

            // note that this container is not `consistent`, in the sense that
            // it does not own the characters pointed to by its strings
            StringLcpContainer<StringSet> quantile_container{
                std::vector<typename StringSet::Char>{},
                {quantile.active().begin(), quantile.active().end()},
                {quantile.lcp(), quantile.lcp() + quantile.size()}};

            auto const quantile_size_chars = quantile.active().get_sum_length();
            this->measuring_tool_.add(quantile_size_chars, "quantile_size");

            this->measuring_tool_.disable();
            Base::sort(quantile_container, comms, quantile_prefixes, builder);
            this->measuring_tool_.enable();
            this->measuring_tool_.stop("sort_globally", "sort_quantile");

            this->measuring_tool_.start("sort_globally", "write_global_permutation");
            auto const ss = quantile_container.make_string_set();
            builder.apply(ss, global_permutation, local_offset, global_offset, comms);
            builder.reset();
            this->measuring_tool_.stop("sort_globally", "write_global_permutation");

            local_offset += size;
            global_offset +=
                comm_root.allreduce_single(kmp::send_buf(size), kmp::op(std::plus<>{}));
            this->measuring_tool_.stop("sort_globally", "quantile_overall", comm_root);
        }

        this->measuring_tool_.setQuantile(0);
        this->measuring_tool_.stop("sort_quantiles", "sort_quantiles_overall");

        return global_permutation;
    }

private:
    static constexpr size_t start_depth = 8;

    size_t quantile_size_;

    template <typename StringPtr, typename ExtraArg>
    std::vector<size_t>
    compute_quantiles(StringPtr const& strptr, ExtraArg const arg, Communicator const& comm) {
        this->measuring_tool_.start("compute_num_quantiles");
        size_t const total_size = sample::accumulate_chars(strptr.active(), arg);
        size_t const local_num_quantiles = tlx::div_ceil(total_size, quantile_size_ + 1);
        size_t const num_quantiles = comm.allreduce_single(
            kamping::send_buf(local_num_quantiles),
            kamping::op(kamping::ops::max<>{})
        );
        this->measuring_tool_.stop("compute_num_quantiles");
        this->measuring_tool_.add(num_quantiles, "num_quantiles");

        this->measuring_tool_.start("compute_quantile_sizes");
        auto const sizes = QuantilePolicy_::compute_partition(strptr, num_quantiles, arg, comm);
        this->measuring_tool_.stop("compute_quantile_sizes");

        return sizes;
    }
};

} // namespace sorter
} // namespace dss_mehnert
