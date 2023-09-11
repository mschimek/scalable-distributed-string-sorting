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

#include "mpi/allgather.hpp"
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

using dss_mehnert::sample::SampleParams;

template <typename Char, bool is_indexed>
using SorterStringSet = std::conditional_t<
    is_indexed,
    dss_schimek::GenericCharLengthIndexStringSet<Char>,
    dss_schimek::GenericCharLengthStringSet<Char>>;

template <bool is_indexed, typename Derived>
class PartitionPolicy {
public:
    template <typename StringPtr, typename Sample>
    static std::vector<size_t> compute_partition(
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

        using StringContainer = dss_schimek::StringContainer<typename Derived::StringSet>;
        StringContainer chosen_splitters;
        if constexpr (is_indexed) {
            auto [splitter_chars, splitter_idxs] =
                get_splitters_indexed(sample_set, num_partitions, comm);
            chosen_splitters = StringContainer{std::move(splitter_chars), splitter_idxs};
        } else {
            auto splitter_chars = get_splitters(sample_set, num_partitions, comm);
            chosen_splitters = StringContainer{std::move(splitter_chars)};
        }
        measuring_tool.stop("choose_splitters");

        measuring_tool.start("compute_interval_sizes");
        auto splitter_set = chosen_splitters.make_string_set();
        std::vector<size_t> interval_sizes;
        if constexpr (is_indexed) {
            // tood this is being computed redundantly multiple times
            auto local_offset = sample::get_local_offset(strptr.size(), comm);
            interval_sizes =
                compute_interval_binary_index(strptr.active(), splitter_set, local_offset);
        } else {
            interval_sizes = compute_interval_binary(strptr.active(), splitter_set);
        }
        measuring_tool.stop("compute_interval_sizes");

        return interval_sizes;
    }
};

template <typename Char, bool is_indexed>
class RQuickV1 : public PartitionPolicy<is_indexed, RQuickV1<Char, is_indexed>> {
    friend PartitionPolicy<is_indexed, RQuickV1<Char, is_indexed>>;

private:
    static constexpr int tag = 50352;
    static constexpr uint64_t seed = 3469931;
    static constexpr bool is_robust = true;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringContainer = dss_schimek::StringContainer<StringSet>;
    using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;

    template <typename Samples>
    static StringContainer sort_samples(Samples&& samples, Communicator const& comm) {
        RQuick2::Comparator<StringPtr> const comp;
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        RQuick::Data<StringContainer, is_indexed> data;
        if constexpr (is_indexed) {
            data = {std::move(samples.sample), {samples.indices}};
        } else {
            data = {std::move(samples), {}};
        }
        return RQuick::sort(gen, std::move(data), MPI_BYTE, tag, comm_mpi, comp, is_robust);
    }
};

template <typename Char, bool is_indexed, bool use_lcps>
class RQuickV2 : public PartitionPolicy<is_indexed, RQuickV2<Char, is_indexed, use_lcps>> {
    friend PartitionPolicy<is_indexed, RQuickV2<Char, is_indexed, use_lcps>>;

private:
    static constexpr int tag = 23560;
    static constexpr uint64_t seed = 3469931;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringPtr = std::conditional_t<
        use_lcps,
        tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>,
        tlx::sort_strings_detail::StringPtr<StringSet>>;

    template <typename Samples>
    static RQuick2::Container<StringPtr> sort_samples(Samples&& samples, Communicator const& comm) {
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        // todo not a huge fan of this
        RQuick2::Data<StringPtr> data;
        if constexpr (is_indexed) {
            data.raw_strs = std::move(samples.sample);
            data.indices = std::move(samples.indices);
        } else {
            data.raw_strs = std::move(samples);
        }
        // LCP array initialization is done by RQuick
        return RQuick2::sort(std::move(data), tag, gen, comm_mpi);
    }
};

// todo create a policy implementation for this
template <typename Sampler, typename StringPtr, typename Params>
typename std::enable_if_t<!Sampler::isIndexed, std::vector<size_t>>
compute_partition_sequential(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto samples = Sampler::sample_splitters(ss, params, comm);
    measuring_tool.stop("sample_splitters");

    measuring_tool.add(samples.size(), "allgather_splitters_bytes_sent");
    measuring_tool.start("allgather_splitters");
    auto splitters = dss_schimek::mpi::allgather_strings(samples, comm);
    measuring_tool.stop("allgather_splitters");

    measuring_tool.start("choose_splitters");
    auto chosen_splitters_cont = choose_splitters(ss, splitters, comm);
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    auto interval_sizes = compute_interval_binary(ss, splitter_set);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

template <typename Sampler, typename StringPtr, typename Params>
typename std::enable_if_t<Sampler::isIndexed, std::vector<size_t>>
compute_partition_sequential(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    using StringSet = dss_schimek::UCharLengthIndexStringSet;
    using StringContainer = dss_schimek::IndexStringLcpContainer<StringSet>;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto sample = Sampler::sample_splitters(ss, params, comm);
    measuring_tool.stop("sample_splitters");

    measuring_tool.add(sample.sample.size(), "allgather_splitters_bytes_sent");
    measuring_tool.start("allgather_splitters");
    auto recv_sample = dss_schimek::mpi::allgatherv(sample.sample, comm);
    auto recv_idxs = dss_schimek::mpi::allgatherv(sample.indices, comm);
    measuring_tool.stop("allgather_splitters");

    measuring_tool.start("choose_splitters");
    StringContainer indexContainer{std::move(recv_sample), recv_idxs};
    indexContainer = choose_splitters(indexContainer, comm);
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = indexContainer.make_string_set();
    auto local_offset = sample::get_local_offset(ss.size(), comm);
    auto interval_sizes = compute_interval_binary_index(ss, splitter_set, local_offset);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

} // namespace partition
} // namespace dss_mehnert
