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
    using Base = BaseDistributedMergeSort<
        Subcommunicators,
        RedistributionPolicy,
        AllToAllStringPolicy,
        SamplePolicy,
        PartitionPolicy>;

    using Base::Base;

    using Char = StringPtr::StringSet::Char;

    using StringPEIndexSet = StringSet<Char, Length, StringIndex, PEIndex>;
    using StringPEIndexPtr = tlx::sort_strings_detail::StringLcpPtr<StringPEIndexSet, size_t>;
    using StringPEIndexContainer = StringLcpContainer<StringPEIndexSet>;

    std::vector<StringIndexPEIndex> sort(
        StringLcpContainer<typename StringPtr::StringSet>&& container, Subcommunicators const& comms
    ) {
        this->measuring_tool_.start("init_container");
        // create a new container with additional rank and string index data members
        std::vector<typename StringPEIndexSet::String> strings;
        strings.reserve(container.size());

        auto const rank = comms.comm_root().rank();
        for (size_t index = 0; auto const& src: container.get_strings()) {
            strings.push_back(src.with_members(StringIndex{index++}, PEIndex{rank}));
        }

        StringLcpContainer<StringPEIndexSet> index_container{
            container.release_raw_strings(),
            std::move(strings),
            container.release_lcps()};
        container.delete_all();
        this->measuring_tool_.stop("init_container");

        return sort(std::move(index_container), comms);
    }

    std::vector<StringIndexPEIndex> sort(
        StringPEIndexContainer&& container,
        Subcommunicators const& comms,
        bool const is_sorted_locally = false
    ) {
        this->measuring_tool_.setPhase("local_sorting");

        auto strptr = container.make_string_lcp_ptr();
        this->measuring_tool_.add(container.char_size(), "chars_in_set");

        if (!is_sorted_locally) {
            this->measuring_tool_.start("local_sorting", "sort_locally");
            tlx::sort_strings_detail::radixsort_CI3(strptr, 0, 0);
            this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());
        } else {
            assert(strptr.active().check_order());
        }

        if (comms.comm_root().size() == 1) {
            return write_permutation(strptr.active());
        } else {
            this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
            BloomFilter bloom_filter{comms, strptr.size()};
            auto const prefixes =
                bloom_filter.compute_distinguishing_prefixes(strptr, comms, start_depth);
            this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

            auto sorted_container = sort_impl_(std::move(container), comms, prefixes);

            this->measuring_tool_.setRound(0);
            return write_permutation(sorted_container.make_string_set());
        }
    }

    std::vector<StringIndexPEIndex> sort(
        StringPEIndexContainer&& container,
        std::vector<size_t> const& dist_prefixes,
        Subcommunicators const& comms
    ) {
        assert(container.make_string_set().check_order());
        auto sorted_container = sort_impl_(std::move(container), comms, dist_prefixes);

        this->measuring_tool_.setRound(0);
        return write_permutation(sorted_container.make_string_set());
    }

private:
    static constexpr uint64_t start_depth = 8;

    template <
        typename Subcommunicators_ = Subcommunicators,
        std::enable_if_t<!Subcommunicators_::is_multi_level, bool> = true>
    auto sort_impl_(
        StringPEIndexContainer&& container,
        Subcommunicators const& comms,
        std::vector<size_t> const& dist_prefixes
    ) {
        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container = this->sort_exhaustive(
            std::move(container),
            comms.comm_final(),
            sample::DistPrefixes{dist_prefixes}
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
        std::vector<size_t> const& dist_prefixes
    ) {
        if (comms.begin() == comms.end()) {
            // special case for single level sort
            this->measuring_tool_.start("sort_globally", "final_sorting");
            auto sorted_container = this->sort_exhaustive(
                std::move(container),
                comms.comm_final(),
                sample::DistPrefixes{dist_prefixes}
            );
            this->measuring_tool_.stop("sort_globally", "final_sorting");

            return sorted_container;
        }

        size_t round = 0;
        auto level = comms.begin();
        {
            // first level of multi-level sort, consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(
                std::move(container),
                *level++,
                sample::DistPrefixes{dist_prefixes}
            );
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        for (; level != comms.end(); ++level) {
            // intermediate level of multi-level sort, don't consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(std::move(container), *level, sample::NoExtraArg{});
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        // final level of multi-level sort, don't consider distinguishing prefixes
        auto sorted_container =
            this->sort_exhaustive(std::move(container), comms.comm_final(), sample::NoExtraArg{});
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        return sorted_container;
    }

    std::vector<StringIndexPEIndex> write_permutation(StringPEIndexSet const& ss) {
        this->measuring_tool_.start("writeback_permutation");
        std::vector<StringIndexPEIndex> permutation(ss.size());

        std::transform(ss.begin(), ss.end(), permutation.begin(), [&ss](auto const& str) {
            return StringIndexPEIndex{str.getStringIndex(), str.getPEIndex()};
        });
        this->measuring_tool_.stop("writeback_permutation");

        this->measuring_tool_.setPhase("none");
        return permutation;
    }
};


namespace _internal {

template <typename StringSet>
StringLcpContainer<StringSet> receive_strings(
    StringLcpContainer<StringSet>&& container,
    std::vector<StringIndexPEIndex> const& permutation,
    Communicator const& comm
) {
    using namespace dss_schimek;
    using MPIRoutine = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;

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
            auto const& str = ss[ss.begin() + recv_requests[offset]];
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
    std::vector<size_t> offsets(comm.size());
    for (auto const& elem: permutation) {
        ++offsets[elem.PEIndex];
    }
    std::exclusive_scan(offsets.begin(), offsets.end(), offsets.begin(), size_t{0});

    // using back_inserter here becuase String may not be default-insertable
    std::vector<typename StringSet::String> strings;
    strings.reserve(ss.size());
    auto op = [&](auto const& x) { return ss[ss.begin() + offsets[x.PEIndex]++]; };
    std::transform(permutation.begin(), permutation.end(), std::back_inserter(strings), op);

    return strings;
}

} // namespace _internal


// todo could also use the gridwise subcommunicators here
// Apply a permutation generated by prefix doubling merge-sort.
template <typename StringSet>
StringLcpContainer<StringSet> apply_permutation(
    StringLcpContainer<StringSet>&& container,
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
