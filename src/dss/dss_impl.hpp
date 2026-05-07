// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <utility>
#include <vector>

#include "mpi/alltoall_strings.hpp"
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
    sorter.sort(container, comms);

    auto strings = container.release_strings();
    std::vector<CharType> out;
    for (auto const& s : strings) {
        out.insert(out.end(), s.getChars(), s.getChars() + s.getLength());
        out.push_back(CharType{0});
    }
    return out;
}
