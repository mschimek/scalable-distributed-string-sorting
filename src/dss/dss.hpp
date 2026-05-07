// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <vector>

#include "mpi/alltoall_strings.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/partition_policy_factory.hpp"

namespace dss {

using SamplerArgs = dss_mehnert::SamplerArgs;
using SplitterSorter = dss_mehnert::SplitterSorter;

inline constexpr dss_mehnert::mpi::AlltoallStringsConfig kDefaultAlltoallConfig{
    .alltoall_kind     = dss_mehnert::mpi::AlltoallvCombinedKind::native,
    .compress_lcps     = true,
    .compress_prefixes = true,
};

inline constexpr SamplerArgs kDefaultSamplerArgs{
    .sample_chars    = false,
    .sample_indexed  = true,
    .sample_random   = false,
    .sampling_factor = 2,
};

template <
    dss_mehnert::mpi::AlltoallStringsConfig AlltoallConfig,
    typename CharType,
    typename Communicator>
std::vector<CharType> run_sorter(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    SamplerArgs const& sampler,
    SplitterSorter splitter_sorter);

template <typename CharType, typename Communicator>
std::vector<CharType> run_sorter(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    SplitterSorter splitter_sorter = SplitterSorter::Sequential)
{
    return run_sorter<kDefaultAlltoallConfig>(
        to_sort, comm, kDefaultSamplerArgs, splitter_sorter);
}

}  // namespace dss

#include "dss/dss_impl.hpp"

namespace dss {
extern template std::vector<unsigned char>
run_sorter<kDefaultAlltoallConfig, unsigned char, dss_mehnert::Communicator>(
    std::vector<unsigned char>&, dss_mehnert::Communicator const&,
    SamplerArgs const&, SplitterSorter);
}
