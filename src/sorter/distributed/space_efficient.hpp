// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <numeric>

#include <kamping/mpi_datatype.hpp>
#include <kamping/named_parameters.hpp>
#include <kamping/p2p/isend.hpp>
#include <kamping/p2p/probe.hpp>
#include <kamping/p2p/recv.hpp>
#include <mpi.h>
#include <tlx/math/div_ceil.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "mpi/rotate.hpp"
#include "sorter/distributed/permutation.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

namespace dss_mehnert {
namespace sorter {
namespace space_efficient {

template <typename CharType, typename Permutation>
class PermutationBuilder;

template <typename CharType>
class PermutationBuilder<CharType, SimplePermutation> {
public:
    using Permutation = SimplePermutation;

    template <PermutationStringSet StringSet, typename Subcommunicators>
    PermutationBuilder(StringSet const&, Subcommunicators const&) {}

    template <PermutationStringSet StringSet>
    void reset(StringSet const&) {}

    template <PermutationStringSet StringSet>
    void push(StringSet const&, std::vector<int>) {}

    template <PermutationStringPtr StringPtr, typename Subcommunicators>
    void apply(
        StringPtr const& strptr, std::span<size_t> global_permutation, Subcommunicators const& comms
    ) {
        Permutation const permutation{strptr.active()};
        permutation.apply(global_permutation, global_offset_, comms);

        global_offset_ += comms.comm_root().allreduce_single(
            kamping::send_buf(strptr.size()),
            kamping::op(std::plus<>{})
        );
    }

private:
    size_t global_offset_ = 0;
};

template <typename CharType>
class PermutationBuilder<CharType, MultiLevelPermutation> {
public:
    using Permutation = MultiLevelPermutation;

    template <PermutationStringSet StringSet, typename Subcommunicators>
    PermutationBuilder(StringSet const&, Subcommunicators const& comms)
        : permutation_(std::distance(comms.begin(), comms.end()) + 1) {}

    template <PermutationStringSet StringSet>
    void reset(StringSet const& ss) {
        current_depth_ = 0;
        permutation_.local().write(ss);
    }

    template <PermutationStringSet StringSet>
    void push(StringSet const& ss, std::vector<int> counts) {
        permutation_.remote(current_depth_++).write(ss, std::move(counts));
    }

    template <PermutationStringPtr StringPtr, typename Subcommunicators>
    void apply(
        StringPtr const& strptr, std::span<size_t> global_permutation, Subcommunicators const& comms
    ) {
        permutation_.apply(global_permutation, global_offset_, comms);
        global_offset_ += comms.comm_root().allreduce_single(
            kamping::send_buf(strptr.size()),
            kamping::op(std::plus<>{})
        );
    }

private:
    size_t current_depth_ = 0;
    size_t global_offset_ = 0;
    MultiLevelPermutation permutation_;
};

template <typename CharType>
class PermutationBuilder<CharType, NonUniquePermutation> {
public:
    using Permutation = NonUniquePermutation;

    template <PermutationStringSet StringSet, typename Subcommunicators>
    PermutationBuilder(StringSet const& ss, Subcommunicators const& comms)
        : permutation_(std::distance(comms.begin(), comms.end()) + 1),
          lower_bound_{0} {}

    template <PermutationStringSet StringSet>
    void reset(StringSet const& ss) {
        current_depth_ = 0;
        permutation_.local().write(ss);
    }

    template <PermutationStringSet StringSet>
    void push(StringSet const& ss, std::vector<int> counts) {
        permutation_.remote(current_depth_++).write(ss, std::move(counts));
    }

    template <PermutationStringPtr StringPtr, typename Subcommunicators>
    void apply(
        StringPtr const& strptr, std::span<size_t> global_permutation, Subcommunicators const& comms
    ) {
        write_offsets(strptr, comms.comm_root());
        permutation_.apply(global_permutation, global_offset_, comms);

        auto const& offsets = permutation_.index_offsets();
        // todo this is being computed/communicated redundantly multiple times
        auto const local_offset_sum = std::accumulate(offsets.begin(), offsets.end(), size_t{0});
        global_offset_ += comms.comm_root().allreduce_single(
            kamping::send_buf(local_offset_sum),
            kamping::op(std::plus<>{})
        );
        is_first_quantile_ = false;
    }

private:
    bool is_first_quantile_ = true;
    size_t current_depth_ = 0;
    size_t global_offset_ = 0;
    NonUniquePermutation permutation_;
    std::vector<CharType> lower_bound_;

    template <PermutationStringPtr StringPtr>
    void write_offsets(StringPtr const& strptr, Communicator const& comm) {
        auto const& ss = strptr.active();
        auto is_equal = [&ss](auto& buf, auto const& rhs) {
            using String = StringPtr::StringSet::String;
            String const lhs{buf.data(), Length{buf.size() - 1}};
            return dss_schimek::calc_lcp(ss, lhs, rhs) == rhs.length;
        };

        auto& index_offsets = permutation_.index_offsets();
        index_offsets.resize(strptr.size());

        // compare first string on PE 0 to last string of previous quantile
        if (!is_first_quantile_ && comm.rank() == 0 && !ss.empty()) {
            index_offsets.front() = !is_equal(lower_bound_, *ss.begin());
        }

        std::vector<CharType> lower_bound{0};
        if (!ss.empty()) {
            auto const& last = *(strptr.active().end() - 1);
            lower_bound.resize(last.length + 1);
            std::copy_n(last.string, last.length, lower_bound.begin());
        } else if (comm.rank() == 0) {
            lower_bound = lower_bound_; // TODO maybe move
        }

        // shift last local string (or string from previous quantile) to the right
        bool const skip_rank = ss.empty() && comm.rank() != 0;
        mpi::rotate_strings_right(lower_bound, lower_bound_, skip_rank, comm);

        size_t first_non_empty_rank = 0;
        if (is_first_quantile_) {
            first_non_empty_rank = comm.allreduce_single(
                kamping::send_buf(ss.empty() ? comm.size() : comm.rank()),
                kamping::op(kamping::ops::min<>{})
            );
        }

        // compare first string to string from (in)directly preceeding PE
        if (!ss.empty() && comm.rank() > first_non_empty_rank) {
            index_offsets.front() = !is_equal(lower_bound_, *ss.begin());
        }

        // compare local strings using LCP values
        if (size_t i = 1; !ss.empty()) {
            for (auto it = ss.begin() + 1; it != ss.end(); ++i, ++it) {
                index_offsets[i] = (strptr.get_lcp(i) != ss[it].length);
            }
        }
    }
};


// this struct is required to disambiguate the PartitionPolicy used
// in sorting and the one use to compute quantiles
template <typename PartitionPolicy>
struct QuantilePolicyBase : public PartitionPolicy {};

template <
    AlltoallStringsConfig config,
    typename RedistributionPolicy,
    typename PartitionPolicy,
    typename QuantilePolicy,
    typename BloomFilter,
    typename Permutation>
class SpaceEfficientSort
    : private QuantilePolicyBase<QuantilePolicy>,
      private prefix_doubling::
          BasePrefixDoublingMergeSort<config, RedistributionPolicy, PartitionPolicy, BloomFilter> {
public:
    using QuantilePolicy_ = QuantilePolicyBase<QuantilePolicy>;
    using Base = prefix_doubling::
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
        using Char = StringSet::Char;

        auto const strptr = container.make_string_lcp_ptr();
        auto const& comm_root = comms.comm_root();

        this->measuring_tool_.setPhase("local_sorting");
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 500 * 1024 * 1024);
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
        PermutationBuilder<Char, Permutation> builder{strptr.active(), comms};

        for (size_t i = 0, local_offset = 0; i != quantile_sizes.size(); ++i) {
            this->measuring_tool_.setQuantile(i);
            this->measuring_tool_.start("sort_globally", "quantile_overall");

            this->measuring_tool_.start("sort_quantiles", "init_permutation");
            auto const size = quantile_sizes[i];
            auto const quantile = strptr.sub(local_offset, size);
            std::span const quantile_prefixes{prefixes.begin() + local_offset, size};
            builder.reset(quantile.active());
            this->measuring_tool_.stop("sort_quantiles", "init_permutation");

            this->measuring_tool_.start("sort_globally", "sort_quantile");
            // note that this container is not `consistent`, in the sense that
            // it does not own the characters pointed to by its strings
            StringLcpContainer<StringSet> quantile_container{
                std::vector<typename StringSet::Char>{},
                {quantile.active().begin(), quantile.active().end()},
                {quantile.lcp(), quantile.lcp() + quantile.size()}};

            auto const quantile_size_chars = quantile.active().get_sum_length();
            this->measuring_tool_.add(quantile_size_chars, "quantile_size");

            // this->measuring_tool_.disable();
            Base::sort(quantile_container, comms, quantile_prefixes, builder);
            // this->measuring_tool_.enable();
            this->measuring_tool_.stop("sort_globally", "sort_quantile");

            this->measuring_tool_.start("sort_globally", "write_global_permutation");
            auto const sorted_ptr = quantile_container.make_string_lcp_ptr();
            builder.apply(sorted_ptr, global_permutation, comms);
            this->measuring_tool_.stop("sort_globally", "write_global_permutation");
            this->measuring_tool_.stop("sort_globally", "quantile_overall");

            local_offset += size;
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

} // namespace space_efficient
} // namespace sorter
} // namespace dss_mehnert
