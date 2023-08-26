// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "mpi/allgather.hpp"
#include "mpi/communicator.hpp"
#include "sorter/RQuick/RQuick.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace partition {

using dss_mehnert::sample::SampleParams;

template <typename Sampler, typename StringPtr, typename Params>
std::enable_if_t<!Sampler::isIndexed, std::vector<size_t>>
compute_partition(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    using namespace dss_schimek;
    using Comparator = StringComparator;
    using StringContainer = StringContainer<UCharLengthStringSet>;
    using RQuickData = RQuick::Data<StringContainer, StringContainer::isIndexed>;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto samples = Sampler::sample_splitters(ss, params, comm);
    measuring_tool.stop("sample_splitters");

    measuring_tool.start("sort_splitters");
    Comparator comp;
    std::mt19937_64 gen{3469931 + comm.rank()};
    RQuickData sample_data{std::move(samples), {}};
    measuring_tool.disable();
    StringContainer sorted_sample = splitterSort(std::move(sample_data), gen, comp, comm);
    measuring_tool.enable();
    measuring_tool.stop("sort_splitters");

    measuring_tool.start("choose_splitters");
    auto chosen_splitters = get_splitters(sorted_sample, params.num_partitions, comm);
    StringContainer chosen_splitters_cont{std::move(chosen_splitters)};
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    auto interval_sizes = compute_interval_binary(ss, splitter_set);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

template <typename Sampler, typename StringPtr, typename Params>
std::enable_if_t<Sampler::isIndexed, std::vector<size_t>>
compute_partition(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    using namespace dss_schimek;
    using Comparator = IndexStringComparator;
    using StringContainer = IndexStringContainer<UCharLengthIndexStringSet>;
    using RQuickData = RQuick::Data<StringContainer, StringContainer::isIndexed>;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuring_tool.start("sample_splitters");
    auto samples = Sampler::sample_splitters(ss, params, comm);
    measuring_tool.stop("sample_splitters");
    // todo remove or move to somewhere else
    measuring_tool.add(samples.sample.size(), "allgather_splitters_bytes_sent");

    measuring_tool.start("sort_splitter");
    Comparator comp;
    std::mt19937_64 gen{3469931 + comm.rank()};
    RQuickData sample_data{std::move(samples.sample), std::move(samples.indices)};
    measuring_tool.disable();
    StringContainer sorted_sample = splitterSort(std::move(sample_data), gen, comp, comm);
    measuring_tool.enable();
    measuring_tool.stop("sort_splitter");

    measuring_tool.start("choose_splitters");
    auto [chosen_splitters, splitter_idxs] =
        get_splitters_indexed(sorted_sample, params.num_partitions, comm);
    StringContainer chosen_splitters_cont{std::move(chosen_splitters), splitter_idxs};
    measuring_tool.stop("choose_splitters");

    measuring_tool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    // tood this is being computed redundantly multiple times
    auto local_offset = sample::get_local_offset(ss.size(), comm);
    auto interval_sizes = compute_interval_binary_index(ss, splitter_set, local_offset);
    measuring_tool.stop("compute_interval_sizes");

    return interval_sizes;
}

template <typename Sampler, typename StringPtr, typename Params>
typename std::enable_if_t<!Sampler::isIndexed, std::vector<size_t>>
compute_partition_sequential(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    auto& measuringTool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuringTool.start("sample_splitters");
    auto samples = Sampler::sample_splitters(ss, params, comm);
    measuringTool.stop("sample_splitters");

    measuringTool.add(samples.size(), "allgather_splitters_bytes_sent");
    measuringTool.start("allgather_splitters");
    auto splitters = dss_schimek::mpi::allgather_strings(samples, comm);
    measuringTool.stop("allgather_splitters");

    measuringTool.start("choose_splitters");
    auto chosen_splitters_cont = choose_splitters(ss, splitters, comm);
    measuringTool.stop("choose_splitters");

    measuringTool.start("compute_interval_sizes");
    auto splitter_set = chosen_splitters_cont.make_string_set();
    auto interval_sizes = compute_interval_binary(ss, splitter_set);
    measuringTool.stop("compute_interval_sizes");

    return interval_sizes;
}

template <typename Sampler, typename StringPtr, typename Params>
typename std::enable_if_t<Sampler::isIndexed, std::vector<size_t>>
compute_partition_sequential(StringPtr string_ptr, Params const& params, Communicator const& comm) {
    using StringSet = dss_schimek::UCharLengthIndexStringSet;
    using StringContainer = dss_schimek::IndexStringLcpContainer<StringSet>;

    auto& measuringTool = measurement::MeasuringTool::measuringTool();
    auto ss = string_ptr.active();

    measuringTool.start("sample_splitters");
    auto sample = Sampler::sample_splitters(ss, params, comm);
    measuringTool.stop("sample_splitters");

    measuringTool.add(sample.sample.size(), "allgather_splitters_bytes_sent");
    measuringTool.start("allgather_splitters");
    auto recv_sample = dss_schimek::mpi::allgatherv(sample.sample, comm);
    auto recv_idxs = dss_schimek::mpi::allgatherv(sample.indices, comm);
    measuringTool.stop("allgather_splitters");

    measuringTool.start("choose_splitters");
    StringContainer indexContainer{std::move(recv_sample), recv_idxs};
    indexContainer = choose_splitters(indexContainer, comm);
    measuringTool.stop("choose_splitters");

    measuringTool.start("compute_interval_sizes");
    auto splitter_set = indexContainer.make_string_set();
    auto local_offset = sample::get_local_offset(ss.size(), comm);
    auto interval_sizes = compute_interval_binary_index(ss, splitter_set, local_offset);
    measuringTool.stop("compute_interval_sizes");

    return interval_sizes;
}

} // namespace partition
} // namespace dss_mehnert
