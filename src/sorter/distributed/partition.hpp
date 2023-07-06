// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include "mpi/communicator.hpp"
#include "sorter/RQuick/RQuick.hpp"
#include "sorter/distributed/misc.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace partition {

template <typename StringPtr, typename Sampler>
std::enable_if_t<!Sampler::isIndexed, std::vector<uint64_t>> compute_partition(
    StringPtr string_ptr,
    uint64_t global_lcp_avg,
    uint64_t num_partitions,
    uint64_t sampling_factor,
    Communicator const& comm
) {
    using namespace dss_schimek;

    using Comparator = StringComparator;
    using StringContainer = StringContainer<UCharLengthStringSet>;
    using RQuickData = RQuick::Data<StringContainer, StringContainer::isIndexed>;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto samples =
        Sampler::sample_splitters(ss, 2 * global_lcp_avg, num_partitions, sampling_factor, comm);
    measuring_tool.stop("sample_splitters");

    measuring_tool.start("sort_splitter");
    Comparator comp;
    std::mt19937_64 gen{3469931 + comm.rank()};
    RQuickData sample_data{std::move(samples), {}};
    measuring_tool.disable(); // todo can these be removed?
    StringContainer sorted_sample = splitterSort(std::move(sample_data), gen, comp, comm);
    measuring_tool.enable();
    measuring_tool.stop("sort_splitter");

    measuring_tool.start("choose_splitters");
    auto chosen_splitters = getSplitters(sorted_sample, num_partitions, comm);
    StringContainer chosen_splitters_cont{std::move(chosen_splitters)};
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    auto interval_sizes = compute_interval_binary(ss, splitter_set);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

template <typename Sampler, typename StringPtr>
std::enable_if_t<Sampler::isIndexed, std::vector<uint64_t>> computePartition(
    StringPtr string_ptr,
    uint64_t global_lcp_avg,
    uint64_t num_partitions,
    uint64_t sampling_factor,
    Communicator const& comm
) {
    using namespace dss_schimek;

    using Comparator = IndexStringComparator;
    using StringContainer = IndexStringContainer<UCharLengthIndexStringSet>;
    using RQuickData = RQuick::Data<StringContainer, StringContainer::isIndexed>;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto samples =
        Sampler::sample_splitters(ss, 2 * global_lcp_avg, num_partitions, sampling_factor);
    measuring_tool.stop("sample_splitters");
    measuring_tool.add(samples.sample.size(), "allgather_splitters_bytes_sent");

    measuring_tool.start("sort_splitter");
    Comparator comp;
    std::mt19937_64 gen{3469931 + comm.rank()};
    RQuickData sample_data{std::move(samples.sample), std::move(samples.indices)};
    measuring_tool.disable(); // todo can these be removed?
    StringContainer sorted_sample = splitterSort(std::move(sample_data), gen, comp, comm);
    measuring_tool.enable();
    measuring_tool.stop("sort_splitter");

    measuring_tool.start("choose_splitters");
    auto [chosen_splitters, splitter_idxs] = getSplittersIndexed(sorted_sample);
    StringContainer chosen_splitters_cont{std::move(chosen_splitters), splitter_idxs};
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    auto local_offset = getLocalOffset(ss.size());
    auto interval_sizes = compute_interval_binary_index(ss, splitter_set, local_offset);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

} // namespace partition
} // namespace dss_mehnert
