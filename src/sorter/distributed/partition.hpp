// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/collectives/allgather.hpp>

#include "mpi/communicator.hpp"
#include "sorter/RQuick/RQuick.hpp"
#include "sorter/RQuick2/RQuick.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace partition {

template <typename Char, bool is_indexed>
using SorterStringSet =
    std::conditional_t<is_indexed, StringSet<Char, Length, Index>, StringSet<Char, Length>>;

template <typename SamplePolicy, typename SplitterPolicy>
class PartitionPolicy : private SamplePolicy, private SplitterPolicy {
public:
    PartitionPolicy() = default;

    explicit PartitionPolicy(size_t const sampling_factor) : SamplePolicy{sampling_factor} {}

    template <typename StringPtr, typename SamplerArg>
    std::vector<size_t> compute_partition(
        StringPtr const& strptr,
        size_t const num_partitions,
        SamplerArg const arg,
        Communicator const& comm
    ) const {
        assert(num_partitions > 0);
        if (num_partitions == 1) {
            return {strptr.size()};
        }

        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("sample_strings");
        auto sample = SamplePolicy::sample_splitters(strptr.active(), num_partitions, arg, comm);
        measuring_tool.stop("sample_strings");

        auto chosen_splitters =
            SplitterPolicy::select_splitters(strptr, std::move(sample), num_partitions, comm);

        measuring_tool.start("compute_intervals");
        auto splitter_set = chosen_splitters.make_string_set();
        std::vector<size_t> interval_sizes;
        if constexpr (SamplePolicy::is_indexed) {
            interval_sizes =
                compute_interval_binary_index(strptr.active(), splitter_set, sample.local_offset);
        } else {
            interval_sizes = compute_interval_binary(strptr.active(), splitter_set);
        }
        measuring_tool.stop("compute_intervals");

        // handle edge cases where less than num_partitions strings are present in total
        assert(interval_sizes.size() <= num_partitions);
        interval_sizes.resize(num_partitions, 0);

        return interval_sizes;
    }
};

template <typename Char, bool is_indexed, typename Derived>
class BaseSplitterPolicy {
public:
    using Sample = sample::SampleResult<Char, is_indexed>;

    template <typename StringPtr>
    static auto select_splitters(
        StringPtr const& strptr,
        Sample&& sample,
        size_t const num_partitions,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("sort_samples");
        auto sorted_sample = Derived::sort_samples(std::move(sample), comm);
        measuring_tool.stop("sort_samples");

        measuring_tool.start("choose_splitters");
        auto sample_set = sorted_sample.make_string_set();
        auto chosen_splitters = Derived::choose_splitters(sample_set, num_partitions, comm);
        measuring_tool.stop("choose_splitters");

        return chosen_splitters;
    }
};

template <typename Char, bool is_indexed>
class RQuickV1 : public BaseSplitterPolicy<Char, is_indexed, RQuickV1<Char, is_indexed>> {
    friend BaseSplitterPolicy<Char, is_indexed, RQuickV1<Char, is_indexed>>;

private:
    static constexpr int tag = 50352;
    static constexpr uint64_t seed = 3469931;
    static constexpr bool is_robust = true;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;
    using Sample = sample::SampleResult<Char, is_indexed>;

    static StringContainer<StringSet> sort_samples(Sample&& sample, Communicator const& comm) {
        RQuick2::Comparator<StringPtr> const comp;
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        RQuick::Data<StringContainer<StringSet>, is_indexed> data;
        if constexpr (is_indexed) {
            data = {std::move(sample.sample), {sample.indices}};
        } else {
            data = {std::move(sample.sample), {}};
        }
        return RQuick::sort(gen, std::move(data), MPI_BYTE, tag, comm_mpi, comp, is_robust);
    }

    static StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const& comm) {
        return choose_splitters_distributed(ss, num_partitions, comm);
    }
};

template <typename Char, bool is_indexed, bool use_lcps>
class RQuickV2 : public BaseSplitterPolicy<Char, is_indexed, RQuickV2<Char, is_indexed, use_lcps>> {
    friend BaseSplitterPolicy<Char, is_indexed, RQuickV2<Char, is_indexed, use_lcps>>;

private:
    static constexpr int tag = 23560;
    static constexpr uint64_t seed = 3469931;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringPtr = std::conditional_t<
        use_lcps,
        tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>,
        tlx::sort_strings_detail::StringPtr<StringSet>>;
    using Sample = sample::SampleResult<Char, is_indexed>;

    static RQuick2::Container<StringPtr> sort_samples(Sample&& sample, Communicator const& comm) {
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        RQuick2::Data<StringPtr> data{std::move(sample.sample)};
        if constexpr (is_indexed) {
            data.indices = std::move(sample.indices);
        }
        // LCP array initialization is done by RQuick
        return RQuick2::sort(std::move(data), tag, gen, comm_mpi);
    }

    static StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const& comm) {
        return choose_splitters_distributed(ss, num_partitions, comm);
    }
};

// todo consider adding a CLI option for this
template <typename Char, bool is_indexed>
class Sequential : public BaseSplitterPolicy<Char, is_indexed, Sequential<Char, is_indexed>> {
    friend BaseSplitterPolicy<Char, is_indexed, Sequential<Char, is_indexed>>;

private:
    using StringSet = SorterStringSet<Char, is_indexed>;
    using Sample = sample::SampleResult<Char, is_indexed>;

    static StringLcpContainer<StringSet> sort_samples(Sample&& sample, Communicator const& comm) {
        auto recv_sample = comm.allgatherv(kamping::send_buf(sample.sample));

        StringLcpContainer<StringSet> global_samples;
        if constexpr (is_indexed) {
            auto recv_indices = comm.allgatherv(kamping::send_buf(sample.indices));
            global_samples = StringContainer{
                recv_sample.extract_recv_buffer(),
                make_initializer<Index>(recv_indices.extract_recv_buffer())};
        } else {
            global_samples = StringLcpContainer{recv_sample.extract_recv_buffer()};
        }

        tlx::sort_strings_detail::radixsort_CI3(global_samples.make_string_lcp_ptr(), 0, 0);
        if constexpr (is_indexed) {
            sort_duplicates(global_samples.make_string_lcp_ptr());
        }

        return global_samples;
    }

    static StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const&) {
        return choose_splitters(ss, num_partitions);
    }
};

} // namespace partition
} // namespace dss_mehnert
