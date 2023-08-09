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

namespace dss_mehnert {
namespace sorter {

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
    using StringPEIndexContainer =
        dss_schimek::StringLcpContainer<dss_schimek::UCharLengthIndexPEIndexStringSet>;

    std::vector<StringIndexPEIndex>
    sort(StringLcpContainer&& container, Subcommunicators const& comms) {
        using namespace kamping;

        auto string_ptr = container.make_string_lcp_ptr();
        auto const& ss = string_ptr.active();
        this->measuring_tool_.setPhase("local_sorting");

        if constexpr (debug) {
            auto op = [&ss](auto sum, auto const& str) { return sum + ss.get_length(str) + 1; };
            size_t chars_in_set = std::accumulate(std::begin(ss), std::end(ss), size_t{0}, op);
            this->measuring_tool_.add(chars_in_set, "chars_in_set");
        }

        this->measuring_tool_.start("local_sorting", "sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        this->measuring_tool_.stop("local_sorting", "sort_locally", comms.comm_root());

        if (comms.comm_root().size() == 1) {
            std::vector<StringIndexPEIndex> result(ss.size());
            auto op = [idx = size_t{0}](auto& x) mutable { x.stringIndex = idx++; };
            std::for_each(result.begin(), result.end(), op);
            return result;
        }

        this->measuring_tool_.start("bloomfilter", "bloomfilter_overall");
        auto prefixes = compute_distinguishing_prefixes(string_ptr, comms.comm_root());
        this->measuring_tool_.stop("bloomfilter", "bloomfilter_overall", comms.comm_root());

        std::tuple<sample::DistPrefixes> sample_args{{prefixes}};
        std::tuple<std::vector<size_t> const&> all_to_all_args{prefixes};

        if (std::begin(comms) == std::end(comms)) {
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

        // note the type of used string container changes from here on out
        StringPEIndexContainer container_PE_idx{std::move(container), comms.comm_root().rank()};

        size_t round = 0;
        {
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            auto level = *std::begin(comms);
            container_PE_idx = this->sort_partial(
                std::move(container_PE_idx),
                level,
                sample_args,
                all_to_all_args
            );
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        for (auto level = std::begin(comms); level != std::end(comms); ++level) {
            this->measuring_tool_.start("sort_globally", "partial_sorting");
            container_PE_idx = this->sort_partial(std::move(container_PE_idx), *level, {}, {});
            this->measuring_tool_.stop("sort_globally", "partial_sorting", comms.comm_root());
            this->measuring_tool_.setRound(++round);
        }

        this->measuring_tool_.start("sort_globally", "final_sorting");
        auto sorted_container =
            this->sort_exhaustive(std::move(container_PE_idx), comms.comm_final(), {}, {});
        this->measuring_tool_.stop("sort_globally", "final_sorting");

        this->measuring_tool_.setRound(0);
        return writeback_permutation(sorted_container);
    }

private:
    static constexpr bool debug = false;
    static constexpr uint64_t start_depth = 8;

    std::vector<StringIndexPEIndex> writeback_permutation(auto& sorted_container) {
        this->measuring_tool_.start("writeback_permutation");

        auto ss = sorted_container.make_string_set();
        std::vector<StringIndexPEIndex> permutation(ss.size());

        auto op = [&ss](auto const& str) -> StringIndexPEIndex {
            return {ss.getIndex(str), ss.getPEIndex(str)};
        };
        std::transform(std::cbegin(ss), std::cend(ss), std::begin(permutation), op);

        this->measuring_tool_.stop("writeback_permutation");
        this->measuring_tool_.setPhase("none");

        return permutation;
    }

    std::vector<size_t>
    compute_distinguishing_prefixes(StringPtr local_string_ptr, Communicator const& comm) {
        using namespace kamping;
        using StringSet = typename StringPtr::StringSet;
        using BloomFilter = dss_schimek::BloomFilter<
            StringSet,
            dss_schimek::FindDuplicates<GolombPolicy>,
            dss_schimek::SendOnlyHashesToFilter<GolombPolicy>,
            dss_schimek::XXHasher>;

        this->measuring_tool_.start("bloomfilter_init");
        StringSet ss = local_string_ptr.active();
        BloomFilter bloom_filter{ss.size(), start_depth};
        std::vector<size_t> results(ss.size());
        this->measuring_tool_.stop("bloomfilter_init");

        size_t current_round = 0;
        this->measuring_tool_.setRound(current_round);
        std::vector<size_t> candidates =
            bloom_filter.filter(local_string_ptr, start_depth, results);

        for (size_t i = (start_depth * 2); i < std::numeric_limits<size_t>::max(); i *= 2) {
            this->measuring_tool_.add(candidates.size(), "bloomfilter_numberCandidates", false);
            this->measuring_tool_.start("bloomfilter_allreduce");
            bool no_candidates = candidates.empty();
            bool all_empty = comm.allreduce(send_buf({no_candidates}), op(ops::logical_and<>{}))
                                 .extract_recv_buffer()[0];
            this->measuring_tool_.stop("bloomfilter_allreduce");

            if (all_empty) {
                break;
            }

            this->measuring_tool_.setRound(++current_round);
            candidates = bloom_filter.filter(local_string_ptr, i, candidates, results);
        }
        this->measuring_tool_.setRound(0);
        return results;
    }
};

} // namespace sorter
} // namespace dss_mehnert
