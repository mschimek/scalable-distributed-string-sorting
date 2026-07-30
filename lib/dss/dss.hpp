// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "dss/mpi/alltoall_strings.hpp"
#include "dss/mpi/communicator.hpp"
#include "dss/sorter/distributed/partition_policy_factory.hpp"

namespace dss {

using SamplerArgs = dss_mehnert::SamplerArgs;
using SplitterSorter = dss_mehnert::SplitterSorter;

inline constexpr dss_mehnert::mpi::AlltoallStringsConfig kDefaultAlltoallConfig{
    .compress_lcps     = true,
    .compress_prefixes = true,
};

inline constexpr SamplerArgs kDefaultSamplerArgs{
    .sample_chars           = false,
    .sample_indexed         = true,
    .sample_random          = false,
    .sampling_factor        = 2,
    .splitter_length_factor = 100,
};

template <
    dss_mehnert::mpi::AlltoallStringsConfig AlltoallConfig,
    typename CharType,
    typename Communicator>
std::vector<CharType> run_sorter(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    SamplerArgs const& sampler,
    SplitterSorter splitter_sorter,
    dss_mehnert::LocalSorter local_sorter = dss_mehnert::LocalSorter::radixsort_CI3,
    dss_mehnert::mpi::AlltoallvParams const& alltoallv_params = {});

template <typename CharType, typename Communicator>
std::vector<CharType> run_sorter(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    SplitterSorter splitter_sorter = SplitterSorter::Sequential,
    dss_mehnert::LocalSorter local_sorter = dss_mehnert::LocalSorter::radixsort_CI3)
{
    return run_sorter<kDefaultAlltoallConfig>(
        to_sort, comm, kDefaultSamplerArgs, splitter_sorter, local_sorter);
}

// Sorts a distributed set of (null-free) strings, packed as a single buffer of
// '\0'-terminated strings, with the LCP-aware hypercube quicksort (RQuick2) as
// the top-level sorter. Returns this PE's share of the globally sorted strings
// in the same packed representation. Note that, as with RQuick, the output is
// distributed across PEs and may be unbalanced (some PEs can end up empty).
template <typename CharType, typename Communicator>
std::vector<CharType> run_rquick(
    std::vector<CharType>& to_sort,
    Communicator const& comm,
    dss_mehnert::LocalSorter local_sorter = dss_mehnert::LocalSorter::radixsort_CI3);

// Indexed variant of the above: each string carries a 64-bit index, and strings
// are ordered lexicographically with the index as a tie-breaker (the same
// indexed RQuick2 configuration used to globally sort splitter candidates).
// `indices` must hold exactly one index per string in `to_sort`. Returns this
// PE's share of the globally sorted strings (packed, '\0'-terminated) together
// with their indices in matching order.
template <typename CharType, typename Communicator>
std::pair<std::vector<CharType>, std::vector<std::uint64_t>> run_rquick(
    std::vector<CharType>& to_sort,
    std::vector<std::uint64_t>& indices,
    Communicator const& comm,
    dss_mehnert::LocalSorter local_sorter = dss_mehnert::LocalSorter::radixsort_CI3);

}  // namespace dss

#include "dss/dss_impl.hpp"

namespace dss {
extern template std::vector<unsigned char>
run_sorter<kDefaultAlltoallConfig, unsigned char, dss_mehnert::Communicator>(
    std::vector<unsigned char>&, dss_mehnert::Communicator const&,
    SamplerArgs const&, SplitterSorter, dss_mehnert::LocalSorter,
    dss_mehnert::mpi::AlltoallvParams const&);

extern template std::vector<unsigned char>
run_rquick<unsigned char, dss_mehnert::Communicator>(
    std::vector<unsigned char>&, dss_mehnert::Communicator const&, dss_mehnert::LocalSorter);

extern template std::pair<std::vector<unsigned char>, std::vector<std::uint64_t>>
run_rquick<unsigned char, dss_mehnert::Communicator>(
    std::vector<unsigned char>&, std::vector<std::uint64_t>&, dss_mehnert::Communicator const&,
    dss_mehnert::LocalSorter);
}
