// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/bcast.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/sort/strings/radix_sort.hpp>

#include "mpi/communicator.hpp"
#include "mpi/environment.hpp"
#include "sorter/RQuick/RQuick.hpp"
#include "sorter/RQuick2/RQuick.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/duplicate_sorting.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringtools.hpp"

namespace dss_schimek {

struct StringComparator {
    using String = dss_schimek::UCharLengthStringSet::String;
    bool operator()(String lhs, String rhs) {
        unsigned char const* lhsChars = lhs.string;
        unsigned char const* rhsChars = rhs.string;
        while (*lhsChars == *rhsChars && *lhsChars != 0) {
            ++lhsChars;
            ++rhsChars;
        }
        return *lhsChars < *rhsChars;
    }
};

struct IndexStringComparator {
    using String = dss_schimek::UCharLengthIndexStringSet::String;
    bool operator()(String lhs, String rhs) {
        unsigned char const* lhsChars = lhs.string;
        unsigned char const* rhsChars = rhs.string;
        while (*lhsChars == *rhsChars && *lhsChars != 0) {
            ++lhsChars;
            ++rhsChars;
        }
        if (*lhsChars == 0 && *rhsChars == 0)
            return lhs.index < rhs.index;
        return *lhsChars < *rhsChars;
    }
};

template <typename StringSet>
IndexStringLcpContainer<StringSet>
choose_splitters(IndexStringLcpContainer<StringSet>& indexContainer, mpi::environment env) {
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    auto all_splitters_strptr = indexContainer.make_string_lcp_ptr();
    tlx::sort_strings_detail::radixsort_CI3(all_splitters_strptr, 0, 0);
    dss_mehnert::sort_duplicates(indexContainer.make_string_lcp_ptr());

    StringSet const& all_splitters_set = all_splitters_strptr.active();
    const size_t nr_splitters = std::min<std::size_t>(env.size() - 1, all_splitters_set.size());
    const size_t splitter_dist = all_splitters_set.size() / (nr_splitters + 1);

    size_t splitterSize = 0u;
    for (std::size_t i = 1; i <= nr_splitters; ++i) {
        auto const begin = all_splitters_set.begin();
        const String splitter = all_splitters_set[begin + i * splitter_dist];
        splitterSize += all_splitters_set.get_length(splitter) + 1;
    }

    std::vector<Char> raw_chosen_splitters(splitterSize);
    std::vector<uint64_t> indices(splitterSize);
    size_t curPos = 0u;
    auto ss = all_splitters_set;

    for (std::size_t i = 1; i <= nr_splitters; ++i) {
        auto const begin = all_splitters_set.begin();
        const String splitter = all_splitters_set[begin + i * splitter_dist];
        indices[i - 1] = splitter.index;
        auto chars = ss.get_chars(splitter, 0);
        const size_t splitterLength = ss.get_length(splitter) + 1;
        std::copy(chars, chars + splitterLength, raw_chosen_splitters.begin() + curPos);
        curPos += splitterLength;
    }
    return IndexStringLcpContainer<StringSet>(std::move(raw_chosen_splitters), indices);
}
template <typename StringSet>
StringLcpContainer<StringSet> choose_splitters(
    StringSet const& ss, std::vector<typename StringSet::Char>& all_splitters, mpi::environment env
) {
    using Char = typename StringSet::Char;
    using String = typename StringSet::String;

    StringLcpContainer<StringSet> all_splitters_cont(std::move(all_splitters));
    tlx::sort_strings_detail::StringLcpPtr all_splitters_strptr =
        all_splitters_cont.make_string_lcp_ptr();
    StringSet const& all_splitters_set = all_splitters_strptr.active();

    tlx::sort_strings_detail::radixsort_CI3(all_splitters_strptr, 0, 0);

    const size_t nr_splitters = std::min<std::size_t>(env.size() - 1, all_splitters_set.size());
    const size_t splitter_dist = all_splitters_set.size() / (nr_splitters + 1);

    size_t splitterSize = 0u;
    for (std::size_t i = 1; i <= nr_splitters; ++i) {
        auto const begin = all_splitters_set.begin();
        const String splitter = all_splitters_set[begin + i * splitter_dist];
        splitterSize += all_splitters_set.get_length(splitter) + 1;
    }

    std::vector<Char> raw_chosen_splitters(splitterSize);
    size_t curPos = 0u;

    for (std::size_t i = 1; i <= nr_splitters; ++i) {
        auto const begin = all_splitters_set.begin();
        const String splitter = all_splitters_set[begin + i * splitter_dist];
        auto chars = ss.get_chars(splitter, 0);
        const size_t splitterLength = ss.get_length(splitter) + 1;
        std::copy(chars, chars + splitterLength, raw_chosen_splitters.begin() + curPos);
        curPos += splitterLength;
    }
    return StringLcpContainer<StringSet>(std::move(raw_chosen_splitters));
}

template <typename StringSet>
inline std::vector<size_t> compute_interval_sizes(
    StringSet const& ss, StringSet const& splitters, dss_schimek::mpi::environment env
) {
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size());

    size_t nr_splitters = std::min<size_t>(env.size() - 1, ss.size());
    size_t splitter_dist = ss.size() / (nr_splitters + 1);
    size_t element_pos = 0;

    for (std::size_t i = 0; i < splitters.size(); ++i) {
        element_pos = (i + 1) * splitter_dist;

        while (element_pos > 0
               && !dss_schimek::leq(
                   ss.get_chars(ss[ss.begin() + element_pos], 0),
                   splitters.get_chars(splitters[splitters.begin() + i], 0)
               )) {
            --element_pos;
        }

        while (element_pos < ss.size()
               && dss_schimek::leq(
                   ss.get_chars(ss[ss.begin() + element_pos], 0),
                   splitters.get_chars(splitters[splitters.begin() + i], 0)
               )) {
            ++element_pos;
        }

        intervals.emplace_back(element_pos);
    }
    intervals.emplace_back(ss.size());
    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

template <typename StringSet>
static inline int binarySearch(StringSet const& ss, typename StringSet::CharIterator elem) {
    using String = typename StringSet::String;

    auto left = ss.begin();
    auto right = ss.end();

    while (left != right) {
        size_t dist = (right - left) / 2;
        String curStr = ss[left + dist];
        int res = dss_schimek::scmp(ss.get_chars(curStr, 0), elem);
        if (res < 0) {
            left = left + dist + 1;
        } else if (res == 0) {
            return left + dist - ss.begin();
        } else {
            right = left + dist;
        }
    }
    return left - ss.begin();
}

inline int indexStringCompare(
    unsigned char const* lhs,
    const uint64_t indexLhs,
    unsigned char const* rhs,
    const uint64_t indexRhs
) {
    while (*lhs == *rhs && *lhs != 0) {
        ++lhs;
        ++rhs;
    } // TODO
    if (*lhs != *rhs) {
        return static_cast<int>(*lhs - *rhs);
    }
    return static_cast<int64_t>(indexLhs - indexRhs);
}

template <typename StringSet>
static inline int binarySearchIndexed(
    StringSet const& ss,
    UCharLengthIndexStringSet const& splitters,
    const uint64_t splitterIndex,
    const uint64_t localOffset
) {
    using String = typename StringSet::String;

    auto left = ss.begin();
    auto right = ss.end();

    while (left != right) {
        size_t dist = (right - left) / 2;
        String curStr = ss[left + dist];
        uint64_t curIndex = left - ss.begin() + dist + localOffset;
        auto splitter = splitters[splitters.begin() + splitterIndex];
        int res = dss_schimek::indexStringCompare(
            ss.get_chars(curStr, 0),
            curIndex,
            splitters.get_chars(splitter, 0),
            splitter.index
        );
        if (res < 0) {
            left = left + dist + 1;
        } else if (res == 0) {
            return left + dist - ss.begin();
        } else {
            right = left + dist;
        }
    }
    return left - ss.begin();
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t>
compute_interval_binary(StringSet const& ss, SplitterSet const& splitters) {
    using CharIt = typename StringSet::CharIterator;
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (std::size_t i = 0; i < splitters.size(); ++i) {
        CharIt splitter = splitters.get_chars(splitters[splitters.begin() + i], 0);
        size_t pos = binarySearch(ss, splitter);
        intervals.emplace_back(pos);
    }
    intervals.emplace_back(ss.size());
    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

template <typename StringSet, typename SplitterSet>
inline std::vector<size_t> compute_interval_binary_index(
    StringSet const& ss, SplitterSet const& splitters, const uint64_t localOffset
) {
    // using CharIt = typename StringSet::CharIterator;
    std::vector<size_t> intervals;
    intervals.reserve(splitters.size() + 1);

    for (std::size_t i = 0; i < splitters.size(); ++i) {
        size_t pos = binarySearchIndexed(ss, splitters, i, localOffset);
        intervals.emplace_back(pos);
    }
    intervals.emplace_back(ss.size());
    std::adjacent_difference(intervals.begin(), intervals.end(), intervals.begin());
    return intervals;
}

} // namespace dss_schimek

namespace dss_mehnert {

namespace _internal {

struct LocalSplitterInterval {
    size_t local_sample_size;
    size_t global_prefix;
    size_t num_splitters;
    size_t splitter_dist;

    bool contains(size_t const global_index) const {
        return minimum() <= global_index && global_index < maximum();
    }

    size_t minimum() const { return global_prefix; }
    size_t maximum() const { return global_prefix + local_sample_size; }

    size_t local_index(size_t const global_index) const { return global_index - global_prefix; }

    template <typename StringSet>
    size_t accumulate_lengths(StringSet const& ss) const {
        using std::begin;

        size_t splitter_size = 0;
        for (size_t i = 1; i <= num_splitters; ++i) {
            size_t const global_index = i * splitter_dist;
            if (contains(global_index)) {
                auto const str = ss[begin(ss) + local_index(global_index)];
                splitter_size += ss.get_length(str) + 1;
            }
        }
        return splitter_size;
    }
};

inline LocalSplitterInterval compute_splitter_interval(
    size_t const local_sample_size, size_t const num_partitions, Communicator const& comm
) {
    size_t const global_prefix = comm.exscan_single(
        kamping::send_buf(local_sample_size),
        kamping::op(kamping::ops::plus<>{})
    );

    size_t total_size = global_prefix + local_sample_size;
    comm.bcast_single(kamping::send_recv_buf(total_size), kamping::root(comm.size() - 1));

    size_t const num_splitters = std::min(num_partitions - 1, total_size);
    size_t const splitter_dist = total_size / (num_splitters + 1);

    return {local_sample_size, global_prefix, num_splitters, splitter_dist};
}

} // namespace _internal

template <typename StringContainer>
std::vector<unsigned char> get_splitters(
    StringContainer& sorted_local_sample, size_t num_partitions, Communicator const& comm
) {
    using std::begin;

    auto splitter_interval =
        _internal::compute_splitter_interval(sorted_local_sample.size(), num_partitions, comm);

    auto const ss = sorted_local_sample.make_string_set();
    auto splitter_size = splitter_interval.accumulate_lengths(ss);
    std::vector<unsigned char> splitter_chars(splitter_size);

    auto dest = splitter_chars.begin();
    for (size_t i = 1; i <= splitter_interval.num_splitters; ++i) {
        size_t const global_idx = i * splitter_interval.splitter_dist;

        if (splitter_interval.contains(global_idx)) {
            auto const local_idx = splitter_interval.local_index(global_idx);
            auto const str = ss[begin(ss) + local_idx];
            auto const length = ss.get_length(str) + 1;
            dest = std::copy_n(ss.get_chars(str, 0), length, dest);
        }
    }

    auto result = comm.allgatherv(kamping::send_buf(splitter_chars));
    return result.extract_recv_buffer();
}

template <typename StringContainer>
std::pair<std::vector<unsigned char>, std::vector<uint64_t>> get_splitters_indexed(
    StringContainer& sorted_local_sample, size_t num_partitions, Communicator const& comm
) {
    using std::begin;

    auto splitter_interval =
        _internal::compute_splitter_interval(sorted_local_sample.size(), num_partitions, comm);

    auto const ss = sorted_local_sample.make_string_set();
    auto splitter_size = splitter_interval.accumulate_lengths(ss);
    std::vector<unsigned char> splitter_chars(splitter_size);
    std::vector<uint64_t> splitter_idxs;

    auto dest = splitter_chars.begin();
    for (size_t i = 1; i <= splitter_interval.num_splitters; ++i) {
        size_t const global_idx = i * splitter_interval.splitter_dist;

        if (splitter_interval.contains(global_idx)) {
            auto const local_idx = splitter_interval.local_index(global_idx);
            auto const str = ss[begin(ss) + local_idx];
            auto const length = ss.get_length(str) + 1;
            dest = std::copy_n(ss.get_chars(str, 0), length, dest);
            splitter_idxs.push_back(str.index);
        }
    }

    auto result_chars = comm.allgatherv(kamping::send_buf(splitter_chars));
    auto result_idxs = comm.allgatherv(kamping::send_buf(splitter_idxs));
    return {result_chars.extract_recv_buffer(), result_idxs.extract_recv_buffer()};
}

template <typename LcpIt>
size_t compute_global_lcp_average(LcpIt const first, LcpIt const last, Communicator const& comm) {
    struct Result {
        size_t lcp_sum, num_strs;

        Result operator+(Result const& rhs) const {
            return {lcp_sum + rhs.lcp_sum, num_strs + rhs.num_strs};
        }
    };

    size_t local_lcp_sum = std::accumulate(first, last, size_t{0});
    size_t num_strs = std::distance(first, last);
    Result local_result{local_lcp_sum, num_strs};

    auto global_result = comm.allreduce_single(
        kamping::send_buf(local_result),
        kamping::op(std::plus<>{}, kamping::ops::commutative)
    );
    return global_result.lcp_sum / std::max(size_t{1}, global_result.num_strs);
}

template <typename Comparator, typename Data>
typename Data::StringContainer
splitter_sort(Data&& data, std::mt19937_64& gen, Comparator& comp, Communicator const& comm) {
    bool const is_robust = true;
    int const tag = 50352;
    auto comm_mpi = comm.mpi_communicator();
    return RQuick::sort(gen, std::forward<Data>(data), MPI_BYTE, tag, comm_mpi, comp, is_robust);
}

template <typename StringPtr>
RQuick2::Container<StringPtr>
splitter_sort_v2(RQuick2::Data<StringPtr>&& data, std::mt19937_64& gen, Communicator const& comm) {
    bool const is_robust = !StringPtr::StringSet::is_indexed;
    int const tag = 23560;
    auto comm_mpi = comm.mpi_communicator();
    return RQuick2::sort(std::move(data), tag, gen, comm_mpi, is_robust);
}

} // namespace dss_mehnert
