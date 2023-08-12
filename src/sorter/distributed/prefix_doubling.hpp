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
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "tlx/die.hpp"

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
    typename AllToAllStringPolicy,
    typename SamplePolicy,
    typename GolombPolicy>
class PrefixDoublingMergeSort
    : private BaseDistributedMergeSort<Subcommunicators, AllToAllStringPolicy, SamplePolicy> {
public:
    using StringLcpContainer = dss_schimek::StringLcpContainer<typename StringPtr::StringSet>;

    using StringPEIndexSet = dss_schimek::UCharLengthIndexPEIndexStringSet;
    using StringPEIndexPtr = tlx::sort_strings_detail::StringLcpPtr<StringPEIndexSet, size_t>;
    using StringPEIndexContainer = dss_schimek::StringLcpContainer<StringPEIndexSet>;

    std::vector<StringIndexPEIndex>
    sort(StringLcpContainer&& container_, Subcommunicators const& comms) {
        using namespace kamping;
        using std::begin;
        using std::end;

        if (comms.comm_root().size() == 1) {
            std::vector<StringIndexPEIndex> result(container_.size());
            for (size_t idx = 0; auto& elem: result) {
                elem.stringIndex = idx++;
            }
            return result;
        }

        this->measuring_tool_.setPhase("local_sorting");

        // todo add proper special case for single level sort
        this->measuring_tool_.start("init_container");
        // create a new container with additional PE rank and string index
        StringPEIndexContainer container{std::move(container_), comms.comm_root().rank()};
        StringPEIndexPtr string_ptr = container.make_string_lcp_ptr();
        StringPEIndexSet const& ss = string_ptr.active();
        this->measuring_tool_.stop("init_container");

        if constexpr (debug) {
            auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str) + 1; };
            size_t chars_in_set = std::accumulate(begin(ss), end(ss), size_t{0}, op);
            this->measuring_tool_.add(chars_in_set, "chars_in_set");
        }

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
        auto prefixes = compute_distinguishing_prefixes(string_ptr, comms.comm_root());
        this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

        std::tuple<sample::DistPrefixes> sample_args{{prefixes}};
        std::tuple<std::vector<size_t> const&> all_to_all_args{prefixes};

        auto level = begin(comms);
        if (level == end(comms)) {
            // special case for single level sort
            this->measuring_tool_.start("sort_globally", "final_sorting");
            auto sorted_container = this->sort_exhaustive(
                std::move(container),
                comms.comm_final(),
                sample_args,
                all_to_all_args
            );
            this->measuring_tool_.stop("sort_globally", "final_sorting");
            return writeback_permutation(sorted_container);
        }

        size_t round = 0;
        {
            // first level of multi-level sort, consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container =
                this->sort_partial(std::move(container), *level++, sample_args, all_to_all_args);
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        for (; level != std::end(comms); ++level) {
            // intermediate level of multi-level sort, don't consider distinguishing prefixes
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container = this->sort_partial(std::move(container), *level, {}, {});
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        // final level of multi-level sort
        auto sorted_container =
            this->sort_exhaustive(std::move(container), comms.comm_final(), {}, {});
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        this->measuring_tool_.setRound(0);
        return writeback_permutation(sorted_container);
    }

private:
    static constexpr bool debug = false;
    static constexpr uint64_t start_depth = 8;

    std::vector<StringIndexPEIndex> writeback_permutation(auto& sorted_container) {
        using std::begin;
        using std::end;

        this->measuring_tool_.start("writeback_permutation");
        auto ss = sorted_container.make_string_set();
        std::vector<StringIndexPEIndex> permutation(ss.size());

        std::transform(begin(ss), end(ss), permutation.begin(), [&ss](auto const& str) {
            return StringIndexPEIndex{ss.getIndex(str), ss.getPEIndex(str)};
        });
        this->measuring_tool_.stop("writeback_permutation");

        this->measuring_tool_.setPhase("none");
        return permutation;
    }

    std::vector<size_t>
    compute_distinguishing_prefixes(StringPEIndexPtr str_ptr, Communicator const& comm) {
        using namespace kamping;
        using BloomFilter = dss_schimek::BloomFilter<
            StringPEIndexSet,
            dss_schimek::FindDuplicates<GolombPolicy>,
            dss_schimek::SendOnlyHashesToFilter<GolombPolicy>,
            dss_schimek::XXHasher>;

        this->measuring_tool_.start("bloomfilter_init");
        auto const& ss = str_ptr.active();
        BloomFilter bloom_filter{ss.size(), start_depth};
        std::vector<size_t> results(ss.size());
        this->measuring_tool_.stop("bloomfilter_init");

        size_t round = 0;
        this->measuring_tool_.setRound(round);
        std::vector<size_t> candidates = bloom_filter.filter(str_ptr, start_depth, results);

        for (size_t i = start_depth * 2; i < std::numeric_limits<size_t>::max(); i *= 2) {
            this->measuring_tool_.add(candidates.size(), "bloomfilter_numberCandidates");
            this->measuring_tool_.start("bloomfilter_allreduce");
            bool no_candidates = candidates.empty();
            bool all_empty = comm.allreduce(send_buf({no_candidates}), op(ops::logical_and<>{}))
                                 .extract_recv_buffer()[0];
            this->measuring_tool_.stop("bloomfilter_allreduce");

            if (all_empty) {
                break;
            }

            this->measuring_tool_.setRound(++round);
            candidates = bloom_filter.filter(str_ptr, i, candidates, results);
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
    dss_schimek::mpi::environment env
) {
    using namespace dss_schimek;
    using MPIRoutine = mpi::AllToAllvCombined<mpi::AllToAllvSmall>;
    using std::begin;

    std::vector<size_t> req_sizes(env.size());
    for (auto const& elem: permutation) {
        ++req_sizes[elem.PEIndex];
    }

    std::vector<size_t> offsets(env.size());
    std::exclusive_scan(req_sizes.begin(), req_sizes.end(), offsets.begin(), size_t{0});

    std::vector<size_t> requests(permutation.size());
    for (auto const& elem: permutation) {
        requests[offsets[elem.PEIndex]++] = elem.stringIndex;
    }

    auto recv_req_sizes = mpi::alltoall(req_sizes, env);
    auto recv_requests = MPIRoutine::alltoallv(requests.data(), req_sizes, env);

    auto const& ss = container.make_string_set();
    std::vector<unsigned char> raw_strs;
    std::vector<size_t> raw_str_sizes(env.size());
    for (size_t rank = 0, offset = 0; rank < env.size(); ++rank) {
        for (size_t i = 0; i < recv_req_sizes[rank]; ++i, ++offset) {
            auto const& str = ss[begin(ss) + recv_requests[offset]];
            auto str_len = ss.get_length(str) + 1;
            auto str_chars = ss.get_chars(str, 0);
            std::copy_n(str_chars, str_len, std::back_inserter(raw_strs));
            raw_str_sizes[rank] += str_len;
        }
    }

    auto recv_chars = MPIRoutine::alltoallv(raw_strs.data(), raw_str_sizes, env);
    return StringLcpContainer<StringSet>{std::move(recv_chars)};
}

template <typename StringSet>
std::vector<typename StringSet::String> reordered_strings(
    StringSet const& ss,
    std::vector<StringIndexPEIndex> const& permutation,
    dss_schimek::mpi::environment env
) {
    using std::begin;

    std::vector<size_t> offsets(env.size());
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
    dss_schimek::mpi::environment env
) {
    auto output_container = _internal::receive_strings(std::move(container), permutation, env);
    assert_equal(output_container.size(), permutation.size());

    auto ss = output_container.make_string_set();
    output_container.set(_internal::reordered_strings(ss, permutation, env));

    return output_container;
}

} // namespace sorter
} // namespace dss_mehnert
