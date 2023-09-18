// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "strings/stringset.hpp"

namespace dss_mehnert {
namespace sorter {

struct StringIndexPEIndex {
    size_t stringIndex;
    size_t PEIndex;

    friend std::ostream& operator<<(std::ostream& stream, StringIndexPEIndex const& indices) {
        return stream << "[" << indices.stringIndex << ", " << indices.PEIndex << "]";
    }
};

template <
    typename StringPtr,
    typename Subcommunicators,
    typename RedistributionPolicy,
    typename AllToAllStringPolicy,
    typename SamplePolicy,
    typename PartitionPolicy,
    typename BloomFilter>
class PrefixDoublingMergeSort : private BaseDistributedMergeSort<
                                    Subcommunicators,
                                    RedistributionPolicy,
                                    AllToAllStringPolicy,
                                    SamplePolicy,
                                    PartitionPolicy> {
public:
    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    using Char = typename StringPtr::StringSet::Char;
    using StringPEIndexSet = dss_schimek::GenericCharLengthIndexPEIndexStringSet<Char>;
    using StringPEIndexPtr = tlx::sort_strings_detail::StringLcpPtr<StringPEIndexSet, size_t>;
    using StringPEIndexContainer = dss_schimek::StringLcpContainer<StringPEIndexSet>;

    std::vector<StringIndexPEIndex>
    sort(StringLcpContainer&& container_, Subcommunicators const& comms) {
        if (comms.comm_root().size() == 1) {
            this->measuring_tool_.start("writeback_permutation");
            std::vector<StringIndexPEIndex> result(container_.size());
            for (size_t idx = 0; auto& elem: result) {
                elem.stringIndex = idx++;
            }
            this->measuring_tool_.stop("writeback_permutation");
            return result;
        } else {
            this->measuring_tool_.start("init_container");
            // create a new container with additional rank and string index data members
            auto container = add_rank_and_index(std::move(container_), comms.comm_root().rank());
            this->measuring_tool_.stop("init_container");

            return sort(std::move(container), comms);
        }
    }


    std::vector<StringIndexPEIndex>
    sort(StringPEIndexContainer&& container, Subcommunicators const& comms) {
        if (comms.comm_root().size() == 1) {
            std::vector<StringIndexPEIndex> result(container.size());
            for (size_t idx = 0; auto& elem: result) {
                elem.stringIndex = idx++;
            }
            return result;
        } else {
            this->measuring_tool_.setPhase("local_sorting");

            auto string_ptr = container.make_string_lcp_ptr();
            this->measuring_tool_.add(container.char_size(), "chars_in_set");

            this->measuring_tool_.start("local_sorting", "sort_locally");
            tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
            this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

            this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
            auto prefixes = compute_distinguishing_prefixes(string_ptr, comms);
            this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

            auto sorted_container = sort_impl_(
                std::move(container),
                comms,
                std::tuple<sample::DistPrefixes>{{prefixes}},
                std::tuple<std::vector<size_t> const&>{prefixes}
            );

            this->measuring_tool_.setRound(0);
            return writeback_permutation(sorted_container);
        }
    }

private:
    static constexpr bool debug = false;
    static constexpr uint64_t start_depth = 8;

    template <
        typename Subcommunicators_ = Subcommunicators,
        std::enable_if_t<!Subcommunicators_::is_multi_level, bool> = true>
    auto sort_impl_(
        StringPEIndexContainer&& container,
        Subcommunicators const& comms,
        auto&& extra_sample_args,
        auto&& extra_all_to_all_args
    ) {
        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container = this->sort_exhaustive(
            std::move(container),
            comms.comm_final(),
            extra_sample_args,
            extra_all_to_all_args
        );
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        return sorted_container;
    }

    template <
        typename Subcommunicators_ = Subcommunicators,
        std::enable_if_t<Subcommunicators_::is_multi_level, bool> = true>
    auto sort_impl_(
        StringPEIndexContainer&& container,
        Subcommunicators const& comms,
        auto&& extra_sample_args,
        auto&& extra_all_to_all_args
    ) {
        using std::begin;
        using std::end;

        if (begin(comms) == end(comms)) {
            // special case for single level sort
            this->measuring_tool_.start("sort_globally", "final_sorting");
            auto sorted_container = this->sort_exhaustive(
                std::move(container),
                comms.comm_final(),
                extra_sample_args,
                extra_all_to_all_args
            );
            this->measuring_tool_.stop("sort_globally", "final_sorting");

            return sorted_container;
        }

        size_t round = 0;
        auto level = begin(comms);
        {
            // first level of multi-level sort, consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(
                std::move(container),
                *level++,
                extra_sample_args,
                extra_all_to_all_args
            );
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        for (; level != end(comms); ++level) {
            // intermediate level of multi-level sort, don't consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(std::move(container), *level, {}, {});
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        // final level of multi-level sort
        auto sorted_container = this->sort_exhaustive(
            std::move(container),
            comms.comm_final(),
            std::tuple<>{},
            std::tuple<>{}
        );
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        return sorted_container;
    }

    std::vector<StringIndexPEIndex> writeback_permutation(auto& sorted_container) {
        using std::begin;
        using std::end;

        this->measuring_tool_.start("writeback_permutation");
        auto ss = sorted_container.make_string_set();
        std::vector<StringIndexPEIndex> permutation(ss.size());

        std::transform(begin(ss), end(ss), permutation.begin(), [&ss](auto const& str) {
            return StringIndexPEIndex{str.getStringIndex(), str.getPEIndex()};
        });
        this->measuring_tool_.stop("writeback_permutation");

        this->measuring_tool_.setPhase("none");
        return permutation;
    }

    StringPEIndexContainer add_rank_and_index(StringLcpContainer&& container, size_t const rank) {
        using dss_schimek::PEIndex;
        using dss_schimek::StringIndex;

        StringPEIndexContainer result;
        result.resize_strings(container.size());

        auto dst = result.getStrings().begin();
        for (size_t index = 0; auto const& src: container.getStrings()) {
            *dst++ = src.with_members(StringIndex{index++}, PEIndex{rank});
        }

        result.set(container.release_raw_strings());
        result.set(container.release_lcps());
        return result;
    }

    std::vector<size_t>
    compute_distinguishing_prefixes(StringPEIndexPtr str_ptr, Subcommunicators const& comms) {
        this->measuring_tool_.start("bloomfilter_init");
        auto const& ss = str_ptr.active();
        BloomFilter bloom_filter{comms, ss.size()};
        std::vector<size_t> results(ss.size());
        this->measuring_tool_.stop("bloomfilter_init");

        size_t round = 0;
        this->measuring_tool_.setRound(round);
        std::vector<size_t> candidates = bloom_filter.filter(str_ptr, start_depth, results);

        for (size_t i = start_depth * 2; i < std::numeric_limits<size_t>::max(); i *= 2) {
            this->measuring_tool_.add(candidates.size(), "bloomfilter_numberCandidates");
            this->measuring_tool_.start("bloomfilter_allreduce");
            auto all_empty = comms.comm_root().allreduce_single(
                kamping::send_buf({candidates.empty()}),
                kamping::op(std::logical_and<>{})
            );
            this->measuring_tool_.stop("bloomfilter_allreduce");

            if (all_empty) {
                break;
            }

            this->measuring_tool_.setRound(++round);
            candidates = bloom_filter.filter(str_ptr, i, results, candidates);
        }

        this->measuring_tool_.setRound(0);
        return results;
    }
};


namespace _internal {

template <typename StringSet>
dss_schimek::StringLcpContainer<StringSet> receive_strings(
    dss_schimek::StringLcpContainer<StringSet>&& container,
    std::vector<StringIndexPEIndex> const& permutation,
    Communicator const& comm
) {
    using namespace dss_schimek;
    using MPIRoutine = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;
    using std::begin;

    std::vector<int> req_sizes(comm.size());
    for (auto const& elem: permutation) {
        ++req_sizes[elem.PEIndex];
    }

    std::vector<size_t> offsets(comm.size());
    std::exclusive_scan(req_sizes.begin(), req_sizes.end(), offsets.begin(), size_t{0});

    std::vector<size_t> requests(permutation.size());
    for (auto const& elem: permutation) {
        requests[offsets[elem.PEIndex]++] = elem.stringIndex;
    }

    auto result = comm.alltoallv(kamping::send_buf(requests), kamping::send_counts(req_sizes));
    auto recv_requests = result.extract_recv_buffer();
    auto recv_req_sizes = result.extract_recv_counts();

    auto const& ss = container.make_string_set();
    std::vector<unsigned char> raw_strs;
    std::vector<size_t> raw_str_sizes(comm.size());
    for (size_t rank = 0, offset = 0; rank < comm.size(); ++rank) {
        for (size_t i = 0; i < static_cast<size_t>(recv_req_sizes[rank]); ++i, ++offset) {
            auto const& str = ss[begin(ss) + recv_requests[offset]];
            auto str_len = ss.get_length(str) + 1;
            auto str_chars = ss.get_chars(str, 0);
            std::copy_n(str_chars, str_len, std::back_inserter(raw_strs));
            raw_str_sizes[rank] += str_len;
        }
    }

    auto recv_chars = MPIRoutine::alltoallv(raw_strs.data(), raw_str_sizes, comm);
    return StringLcpContainer<StringSet>{std::move(recv_chars)};
}

template <typename StringSet>
std::vector<typename StringSet::String> reordered_strings(
    StringSet const& ss,
    std::vector<StringIndexPEIndex> const& permutation,
    Communicator const& comm
) {
    using std::begin;

    std::vector<size_t> offsets(comm.size());
    for (auto const& elem: permutation) {
        ++offsets[elem.PEIndex];
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

    // using back_inserter here becuase String may not be default-insertable
    std::vector<typename StringSet::String> strings;
    strings.reserve(ss.size());
    auto op = [&](auto const& x) { return ss[begin(ss) + offsets[x.PEIndex]++]; };
    std::transform(permutation.begin(), permutation.end(), std::back_inserter(strings), op);

    return strings;
}

} // namespace _internal


// todo could also use the gridwise subcommunicators here
// Apply a permutation generated by prefix doubling merge-sort.
template <typename StringSet>
dss_schimek::StringLcpContainer<StringSet> apply_permutation(
    dss_schimek::StringLcpContainer<StringSet>&& container,
    std::vector<StringIndexPEIndex> const& permutation,
    Communicator const& comm
) {
    auto output_container = _internal::receive_strings(std::move(container), permutation, comm);
    assert_equal(output_container.size(), permutation.size());

    auto ss = output_container.make_string_set();
    output_container.set(_internal::reordered_strings(ss, permutation, comm));

    return output_container;
}

} // namespace sorter
} // namespace dss_mehnert
