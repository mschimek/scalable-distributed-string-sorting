// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall_strings.hpp"
#include "sorter/RQuick2/RQuick.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"
#include "sorter/distributed/redistribution.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"

template <
    dss_mehnert::mpi::AlltoallStringsConfig AlltoallConfig,
    typename CharType,
    typename Communicator>
std::vector<CharType> dss::run_sorter(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    SamplerArgs const& sampler,
    SplitterSorter splitter_sorter)
{
    using StringSet            = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using PartitionPolicy      = dss_mehnert::MergeSortPartitionPolicy<CharType>;
    using RedistributionPolicy =
        dss_mehnert::redistribution::NoRedistribution<dss_mehnert::mpi::Communicator>;
    using Subcommunicators     = typename RedistributionPolicy::Subcommunicators;
    using MergeSort            = dss_mehnert::sorter::
        DistributedMergeSort<AlltoallConfig, RedistributionPolicy, PartitionPolicy>;

    Subcommunicators comms{comm};
    MergeSort sorter{
        dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(sampler, splitter_sorter),
        RedistributionPolicy{},
    };
    dss_mehnert::StringLcpContainer<StringSet> container{std::move(to_sort)};
    sorter.sort(container, comms, sampler.splitter_length_factor);

    auto strings = container.release_strings();
    std::vector<CharType> out;
    for (auto const& s : strings) {
        out.insert(out.end(), s.getChars(), s.getChars() + s.getLength());
        out.push_back(CharType{0});
    }
    return out;
}

template <typename CharType, typename Communicator>
std::pair<std::vector<CharType>, std::vector<std::uint64_t>> dss::run_rquick(
    std::vector<CharType>& to_sort,
    std::vector<std::uint64_t>& indices,
    Communicator const& comm)
{
    using StringSet =
        dss_mehnert::StringSet<CharType, dss_mehnert::Length, dss_mehnert::Index>;
    using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    // Arbitrary but fixed; only needs to be unique among concurrent RQuick calls.
    constexpr int tag       = 47120;
    constexpr uint64_t seed = 3469931;

    std::mt19937_64 gen{seed + comm.rank()};
    auto const comm_mpi = comm.mpi_communicator();

    // LCP array initialization is handled by RQuick.
    RQuick2::Data<StringPtr> data{std::move(to_sort)};
    data.indices = std::move(indices);
    auto container = RQuick2::sort(std::move(data), tag, gen, comm_mpi);

    auto strings = container.release_strings();
    std::vector<CharType> out_chars;
    std::vector<std::uint64_t> out_indices;
    out_indices.reserve(strings.size());
    for (auto const& s : strings) {
        out_chars.insert(out_chars.end(), s.getChars(), s.getChars() + s.getLength());
        out_chars.push_back(CharType{0});
        out_indices.push_back(s.getIndex());
    }
    return {std::move(out_chars), std::move(out_indices)};
}

template <typename CharType, typename Communicator>
std::vector<CharType> dss::run_rquick(std::vector<CharType>& to_sort, Communicator const& comm)
{
    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using StringPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;

    // Arbitrary but fixed; only needs to be unique among concurrent RQuick calls.
    constexpr int tag       = 47110;
    constexpr uint64_t seed = 3469931;

    std::mt19937_64 gen{seed + comm.rank()};
    auto const comm_mpi = comm.mpi_communicator();

    // LCP array initialization is handled by RQuick.
    RQuick2::Data<StringPtr> data{std::move(to_sort)};
    auto container = RQuick2::sort(std::move(data), tag, gen, comm_mpi);

    auto strings = container.release_strings();
    std::vector<CharType> out;
    for (auto const& s : strings) {
        out.insert(out.end(), s.getChars(), s.getChars() + s.getLength());
        out.push_back(CharType{0});
    }
    return out;
}
