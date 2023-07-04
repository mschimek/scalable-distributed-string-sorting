#pragma once

#include <algorithm>
#include <random>
#include <type_traits>

#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "merge/bingmann-lcp_losertree.hpp"
#include "merge/stringtools.hpp"
#include "mpi/allgather.hpp"
#include "mpi/alltoall.hpp"
#include "mpi/byte_encoder.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/synchron.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/samplingStrategies.hpp"
#include "sorter/distributed/tracker.hpp"
#include "strings/stringptr.hpp"
#include "strings/stringset.hpp"
#include "strings/stringtools.hpp"
#include "util/measuringTool.hpp"
#include "util/structs.hpp"

namespace dss_schimek {

// DEGUB FUNCTION
template <typename StringPtr>
std::vector<size_t> computeResultsWithChecks(StringPtr local_string_ptr) {
    using StringSet = typename StringPtr::StringSet;
    dss_schimek::mpi::environment env;

    StringSet ss = local_string_ptr.active();
    size_t sum = 0;
    for (auto const& str: ss) {
        auto const length = ss.get_length(str) + 1;
        auto const chars = ss.get_chars(str, 0);
        for (size_t i = 0; i < length; ++i)
            sum += *chars;
    }

    std::vector<size_t> total_sum = dss_schimek::mpi::allgather(sum, env);
    std::cout << "TotalSum: "
              << std::accumulate(total_sum.begin(), total_sum.end(), static_cast<size_t>(0u))
              << std::endl;
    std::vector<size_t> results(ss.size(), 0);
    std::vector<size_t> candidates(ss.size());
    std::iota(candidates.begin(), candidates.end(), 0);
    std::vector<size_t> results_exact(ss.size(), 0);
    std::vector<size_t> candidates_exact(ss.size());
    std::iota(candidates_exact.begin(), candidates_exact.end(), 0);
    std::vector<size_t> results_tracker;
    std::vector<size_t> candidates_tracker;
    const size_t startDepth = 4u;

    BloomFilter<
        StringSet,
        FindDuplicates<AllToAllHashesNaive>,
        SendOnlyHashesToFilter<AllToAllHashesNaive>,
        SipHasher>
        bloomFilter(ss.size(), startDepth);
    ExcatDistinguishingPrefix<StringSet> exactDistinguishingPrefixAlgo;
    exactDistinguishingPrefixAlgo.filter_exact(local_string_ptr, candidates_exact, results_exact);

    Tracker<StringSet> tracker(std::numeric_limits<uint32_t>::max());
    tracker.init(local_string_ptr);

    std::cout << "tracker init rank: " << env.rank() << std::endl;
    std::cout << "filter_exact rank: " << env.rank() << std::endl;
    size_t curIteration = 0;

    for (size_t i = startDepth; i < std::numeric_limits<size_t>::max(); i *= 2) {
        env.barrier();
        if (env.rank() == 0)
            std::cout << "\t\t\t\t\t curIteration: " << i << std::endl;
        candidates = bloomFilter.filter(local_string_ptr, i, candidates, results);
        tracker.startIteration(i);
        results_tracker = tracker.getResultsOf(env.rank());
        candidates_tracker = tracker.getCandidatesOf(env.rank());
        dss_schimek::mpi::execute_in_order(
            [&]() {
                sort(candidates.begin(), candidates.end());
                sort(candidates_tracker.begin(), candidates_tracker.end());
                if (candidates != candidates_tracker) {
                    size_t smallerSize = std::min(candidates.size(), candidates_tracker.size());
                    std::cout << "rank: " << env.rank() << std::endl;
                    for (size_t i = 0; i < smallerSize; ++i) {
                        std::cout << i << " candidate: " << candidates[i]
                                  << " candidates_tracker: " << candidates_tracker[i] << std::endl;
                        if (candidates[i] != candidates_tracker[i]) {
                            tracker.printDuplicateWitness(candidates_tracker[i], env.rank());
                            std::abort();
                        }
                    }
                    std::cout << "candidates sets differ!" << std::endl;
                    std::abort();
                }
                if (results != results_tracker) {
                    std::cout << "rank: " << env.rank() << std::endl;
                    for (size_t i = 0; i < results.size(); ++i) {
                        std::cout << i << " results " << results[i] << " results_tracker "
                                  << results_tracker[i] << std::endl;
                        if (results[i] != results_tracker[i])
                            std::abort();
                    }
                }
            },
            env
        );

        bool noMoreCandidates = candidates.empty();
        std::cout << i << " rank: " << env.rank() << " candiates " << candidates.size()
                  << " noMoreCandiates: " << noMoreCandidates << std::endl;
        bool allEmpty = dss_schimek::mpi::allreduce_and(noMoreCandidates, env);
        std::cout << "allEmpty?: " << allEmpty << std::endl;
        if (allEmpty)
            break;
        ++curIteration;
    }

    // size_t diffcount = 0;
    // size_t diffsum = 0;
    // std::vector<size_t> histogram(100, 0);
    ////std::cout << "compare results: rank: " << env.rank() << std::endl;
    // dss_schimek::mpi::execute_in_order([&]() {
    //    std::cout << "rank: " << env.rank() << std::endl;
    //    for (size_t i = 0; i < ss.size(); ++i) {
    //    std::cout << i << " " << results[i] << " " << results_tracker[i] << "
    //    " << results_exact[i] << std::endl; if ( results[i] !=
    //    results_tracker[i] || results[i] < results_exact[i]) {
    //    tracker.printDuplicateWitness(i, env.rank());
    //    std::abort();
    //    }
    //    else if(results[i] > results_exact[i]) {
    //    size_t diff = results[i] - results_exact[i];
    //    if (diff < histogram.size())
    //    histogram[diff]++;
    //    diffsum += diff;
    //    ++diffcount;
    //    }
    //    }});
    // std::cout << "diffsum: " << diffsum << " diffcount: " << diffcount << "
    // avg diff: " << static_cast<double>(diffsum) / diffcount << std::endl;
    return results;
}

template <typename StringPtr, typename GolombPolicy>
std::vector<size_t> computeDistinguishingPrefixes(StringPtr local_string_ptr) {
    using StringSet = typename StringPtr::StringSet;
    using namespace dss_schimek::measurement;
    using Hasher = XXHasher;

    const uint64_t startDepth = 8;

    dss_schimek::mpi::environment env;
    MeasuringTool& measuringTool = MeasuringTool::measuringTool();

    measuringTool.disable();
    measuringTool.start(std::string("bloomfilter_init"));
    StringSet ss = local_string_ptr.active();
    BloomFilter<
        StringSet,
        FindDuplicates<GolombPolicy>,
        SendOnlyHashesToFilter<GolombPolicy>,
        Hasher>
        bloomFilter(ss.size(), startDepth);
    std::vector<size_t> results(ss.size(), 0);
    measuringTool.stop(std::string("bloomfilter_init"));

    size_t curIteration = 0;
    measuringTool.setRound(curIteration);
    std::vector<size_t> candidates = bloomFilter.filter(local_string_ptr, startDepth, results);

    for (size_t i = (startDepth * 2); i < std::numeric_limits<size_t>::max(); i *= 2) {
        measuringTool.add(candidates.size(), std::string("bloomfilter_numberCandidates"), false);
        measuringTool.start(std::string("bloomfilter_allreduce"));
        bool noMoreCandidates = candidates.empty();
        bool allEmpty = dss_schimek::mpi::allreduce_and(noMoreCandidates, env);
        measuringTool.stop(std::string("bloomfilter_allreduce"));
        if (allEmpty)
            break;

        measuringTool.setRound(++curIteration);
        candidates = bloomFilter.filter(local_string_ptr, startDepth + i, candidates, results);
    }
    measuringTool.enable();
    measuringTool.setRound(0);
    return results;
}

template <
    typename StringPtr,
    typename SampleSplittersPolicy,
    typename AllToAllStringPolicy,
    typename GolombEncoding>
class DistributedPrefixDoublingSort : private AllToAllStringPolicy {
public:
    std::vector<StringIndexPEIndex> sort(
        dss_schimek::StringLcpContainer<typename StringPtr::StringSet>&& container,
        StringPtr& local_string_ptr,
        dss_schimek::mpi::environment env
    ) {
        // constexpr bool debug = false;
        using dss_schimek::measurement::MeasuringTool;

        using StringSet = typename StringPtr::StringSet;
        // using Char = typename StringSet::Char;

        MeasuringTool& measuringTool = MeasuringTool::measuringTool();
        StringSet const& ss = local_string_ptr.active();
        measuringTool.setPhase("local_sorting");

        size_t charactersInSet = 0;
        for (auto const& str: ss) {
            charactersInSet += ss.get_length(str) + 1;
        }

        measuringTool.add(charactersInSet, "charactersInSet");
        // sort locally
        measuringTool.start("sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(local_string_ptr, 0, 0);
        measuringTool.stop("sort_locally");

        // measuringTool.start("avg_lcp");
        // const size_t globalLcpAvg = getAvgLcp(local_string_ptr) + lcpSummand;
        // measuringTool.stop("avg_lcp");

        // There is only one PE, hence there is no need for distributed sorting
        if (env.size() == 1)
            return std::vector<StringIndexPEIndex>();

        measuringTool.setPhase("bloomfilter");
        measuringTool.start("bloomfilter_overall");
        // measuringTool.disableMeasurement();
        std::vector<size_t> results =
            computeDistinguishingPrefixes<StringPtr, GolombEncoding>(local_string_ptr);
        // measuringTool.enableMeasurement();
        measuringTool.stop("bloomfilter_overall");
        measuringTool.setPhase("bucket_computation");

        auto interval_sizes =
            computePartition<SampleSplittersPolicy, StringPtr>(local_string_ptr, 2, results);

        measuringTool.setPhase("string_exchange");
        measuringTool.start("all_to_all_strings");
        std::vector<std::size_t> receiving_interval_sizes =
            dss_schimek::mpi::alltoall(interval_sizes);
        dss_schimek::StringLcpContainer<UCharIndexPEIndexStringSet> recv_string_cont =
            AllToAllStringPolicy::alltoallv(std::move(container), interval_sizes, results);
        container.deleteAll();
        measuringTool.stop("all_to_all_strings");

        measuringTool.setPhase("merging");

        measuringTool.add(
            recv_string_cont.char_size() - recv_string_cont.size(),
            "num_received_chars",
            false
        );
        measuringTool.add(recv_string_cont.size(), "num_recv_strings", false);
        size_t num_recv_elems = std::accumulate(
            receiving_interval_sizes.begin(),
            receiving_interval_sizes.end(),
            static_cast<size_t>(0u)
        );

        measuringTool.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(
                recv_string_cont,
                receiving_interval_sizes,
                env
            );
        measuringTool.stop("compute_ranges");

        measuringTool.start("merge_ranges");
        auto sorted_container =
            choose_merge<AllToAllStringPolicy>(std::move(recv_string_cont), ranges, num_recv_elems);
        measuringTool.stop("merge_ranges");
        measuringTool.start("writeback_permutation");
        auto sortedSet = sorted_container.make_string_set();
        std::vector<StringIndexPEIndex> permutation;
        permutation.reserve(sortedSet.size());

        for (auto str: sortedSet)
            permutation.emplace_back(sortedSet.getIndex(str), sortedSet.getPEIndex(str));
        measuringTool.stop("writeback_permutation");
        measuringTool.setPhase("none");

        return permutation;
    }
};
} // namespace dss_schimek
