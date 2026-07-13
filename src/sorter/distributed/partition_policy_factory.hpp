// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include <tlx/die/core.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "options.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/partition_long_filter.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringset.hpp"

namespace dss_mehnert {

enum class SplitterSorter { RQuickV1, RQuickV2, RQuickLcp, RQuickLongFilter, Sequential };

struct SamplerArgs {
    bool sample_chars = false;
    bool sample_indexed = false;
    bool sample_random = false;
    size_t sampling_factor = 2;
    // Maximum splitter length is derived as splitter_length_factor * (avg_lcp + 5);
    // a larger factor allows longer (more discriminating) splitters at the cost of
    // larger samples.
    size_t splitter_length_factor = 100;
    // Pseudorandomly redistribute the splitter sample across the PEs before it is
    // sorted, so RQuick's per-string-count balancing starts from a balanced sample.
    bool redistribute_sample = false;
    // The imbalance bounds of the levels multiply, so with multiple levels the sampling
    // factor is scaled by the number of levels to keep the *final* imbalance at the bound
    // a single level would give.
    bool level_adjusted_scaling = false;

    SamplerArgs scaled_to_levels(size_t const num_levels) const {
        SamplerArgs scaled = *this;
        if (level_adjusted_scaling) {
            scaled.sampling_factor *= std::max<size_t>(1, num_levels);
        }
        return scaled;
    }
};

[[noreturn]] inline void die_with_feature(std::string_view feature) {
    tlx_die("feature disabled for compile time; enable with '-D" << feature << "=On'");
}

namespace redistribution {

template <typename StringSet, typename Subcommunicators>
class PolymorphicRedistributionPolicy
    : public RedistributionBase<
          Subcommunicators,
          PolymorphicRedistributionPolicy<StringSet, Subcommunicators>> {
public:
    using Communicator = Subcommunicators::Communicator;

    template <typename RedistributionPolicy>
    explicit PolymorphicRedistributionPolicy(RedistributionPolicy policy)
        : self_{new RedistributionObject<RedistributionPolicy>{std::move(policy)}} {}

    template <typename Strings>
    std::vector<size_t> impl(
        Strings const& strings,
        std::vector<size_t> const& intervals,
        Level<Communicator> const& level
    ) const {
        return self_->impl(strings, intervals, level);
    }

private:
    struct RedistributionConcept {
        virtual ~RedistributionConcept() = default;

        virtual std::vector<size_t> impl(
            FullStrings<StringSet> const& strings,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const = 0;

        virtual std::vector<size_t> impl(
            Prefixes const& prefixes,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const = 0;
    };

    template <typename RedistributionPolicy>
    struct RedistributionObject : public RedistributionConcept, private RedistributionPolicy {
        explicit RedistributionObject(RedistributionPolicy policy)
            : RedistributionPolicy{std::move(policy)} {}

        virtual std::vector<size_t> impl(
            FullStrings<StringSet> const& strings,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const override {
            return RedistributionPolicy::impl(strings, intervals, level);
        }

        virtual std::vector<size_t> impl(
            Prefixes const& prefixes,
            std::vector<size_t> const& intervals,
            Level<Communicator> const& level
        ) const override {
            return RedistributionPolicy::impl(prefixes, intervals, level);
        }
    };

    std::unique_ptr<RedistributionConcept> self_;
};

} // namespace redistribution

template <typename StringSet, typename... SamplerArgs>
class PolymorphicPartitionPolicy {
public:
    using This = PolymorphicPartitionPolicy<StringSet, SamplerArgs...>;
    using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    template <typename PartitionPolicy>
    explicit PolymorphicPartitionPolicy(PartitionPolicy policy)
        : self_{new PartitionObject<PartitionPolicy>{std::move(policy)}} {}

    template <typename SamplerArg>
    std::vector<size_t> compute_partition(
        StringPtr const& strptr,
        size_t const num_partitions,
        SamplerArg const arg,
        Communicator const& comm
    ) const {
        return self_->compute_partition(strptr, num_partitions, arg, comm);
    }

private:
    template <typename SamplerArg>
    struct PartitionConcept_ {
        virtual ~PartitionConcept_() = default;

        virtual std::vector<size_t> compute_partition(
            StringPtr const& strptr,
            size_t const num_partitions,
            SamplerArg const arg,
            Communicator const& comm
        ) const = 0;
    };

    struct PartitionConcept : public PartitionConcept_<SamplerArgs>... {
        using PartitionConcept_<SamplerArgs>::compute_partition...;
    };

    template <typename PartitionPolicy, typename SamplerArg>
    struct PartitionObject_ : public virtual PartitionConcept, private virtual PartitionPolicy {
        virtual std::vector<size_t> compute_partition(
            StringPtr const& strptr,
            size_t const num_partitions,
            SamplerArg const arg,
            Communicator const& comm
        ) const override {
            return PartitionPolicy::compute_partition(strptr, num_partitions, arg, comm);
        }
    };

    template <typename PartitionPolicy>
    struct PartitionObject final : public PartitionObject_<PartitionPolicy, SamplerArgs>... {
        explicit PartitionObject(PartitionPolicy policy) : PartitionPolicy{std::move(policy)} {}
    };

    std::unique_ptr<PartitionConcept> self_;
};

template <typename Char>
using MergeSortPartitionPolicy =
    PolymorphicPartitionPolicy<StringSet<Char, Length>, sample::MaxLength>;

template <typename Char, typename LengthType, typename Permutation>
using PrefixDoublingPartitionPolicy = PolymorphicPartitionPolicy<
    sorter::AugmentedStringSet<StringSet<Char, LengthType>, Permutation>,
    sample::NoExtraArg,
    sample::DistPrefixes>;

template <typename Char, typename LengthType, typename Permutation>
using SpaceEfficientPartitionPolicy = PolymorphicPartitionPolicy<
    sorter::AugmentedStringSet<CompressedStringSet<Char, LengthType>, Permutation>,
    sample::NoExtraArg,
    sample::MaxLength,
    sample::DistPrefixes>;

template <typename Char, typename PolymorphicPolicy>
PolymorphicPolicy
init_partition_policy(SamplerArgs const& sampler, SplitterSorter splitter_sorter) {
    auto dispatch_policy = [&]<typename PartitionPolicy> {
        return PolymorphicPolicy{
            PartitionPolicy{sampler.sampling_factor, sampler.redistribute_sample}};
    };

    auto dispatch_sorter = [&]<typename SamplePolicy> {
        using namespace dss_mehnert::partition;

        constexpr bool indexed = SamplePolicy::is_indexed;

        switch (splitter_sorter) {
            case SplitterSorter::RQuickV1: {
                if constexpr (CliOptions::enable_rquick_v1) {
                    using SplitterPolicy = RQuickV1<Char, indexed>;
                    using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                    return dispatch_policy.template operator()<PartitionPolicy>();
                } else {
                    die_with_feature("CLI_ENABLE_RQUICK_V1");
                }
            }
            case SplitterSorter::RQuickV2: {
                using SplitterPolicy = RQuickV2<Char, indexed, false>;
                using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                return dispatch_policy.template operator()<PartitionPolicy>();
            }
            case SplitterSorter::RQuickLcp: {
                if constexpr (CliOptions::enable_rquick_lcp) {
                    using SplitterPolicy = RQuickV2<Char, indexed, true>;
                    using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                    return dispatch_policy.template operator()<PartitionPolicy>();
                } else {
                    die_with_feature("CLI_ENABLE_RQUICK_LCP");
                }
            }
            case SplitterSorter::RQuickLongFilter: {
                if constexpr (CliOptions::enable_rquick_long_filter) {
                    // the long filter overloads the Index slot with per-long
                    // ranks, so it requires indexed sampling (Increment 1/2)
                    if constexpr (indexed) {
                        using SplitterPolicy = RQuickV2LongFilter<Char, indexed, false>;
                        using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                        return dispatch_policy.template operator()<PartitionPolicy>();
                    } else {
                        tlx_die("the RQuick long filter requires indexed sampling ('-I')");
                    }
                } else {
                    die_with_feature("CLI_ENABLE_RQUICK_LONG_FILTER");
                }
            }
            case SplitterSorter::Sequential: {
                using SplitterPolicy = Sequential<Char, indexed>;
                using PartitionPolicy = PartitionPolicy<SamplePolicy, SplitterPolicy>;
                return dispatch_policy.template operator()<PartitionPolicy>();
            }
        }
        tlx_die("unknown splitter sorter");
    };

    auto dispatch_sampler = [&]<bool indexed, bool random> {
        using namespace dss_mehnert::sample;

        if (sampler.sample_chars) {
            using SamplePolicy = CharBasedSampling<indexed, random>;
            return dispatch_sorter.template operator()<SamplePolicy>();
        } else {
            using SamplePolicy = StringBasedSampling<indexed, random>;
            return dispatch_sorter.template operator()<SamplePolicy>();
        }
    };

    auto dispatch_random = [&]<bool indexed> {
        if (sampler.sample_random) {
            return dispatch_sampler.template operator()<indexed, true>();
        } else {
            return dispatch_sampler.template operator()<indexed, false>();
        }
    };

    if (sampler.sample_indexed) {
        return dispatch_random.template operator()<true>();
    } else {
        return dispatch_random.template operator()<false>();
    }
}

} // namespace dss_mehnert
