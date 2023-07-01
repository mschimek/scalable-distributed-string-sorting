// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <random>
#include <type_traits>

#include <kamping/communicator.hpp>
#include <tlx/container/loser_tree.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/misc.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"

namespace dss_schimek {

static constexpr bool debug = false;

struct string_compare {
    using String = dss_schimek::UCharLengthStringSet::String;
    bool operator()(String a, String b) {
        auto charsA = a.string;
        auto charsB = b.string;
        while (*charsA == *charsB && *charsA != 0) {
            ++charsA;
            ++charsB;
        }
        return *charsA < *charsB;
    }
};

template <typename StringContainer, typename Ranges>
StringContainer noLcpMerge(StringContainer&& stringContainer, Ranges const& ranges) {
    dss_schimek::mpi::environment env;
    using StringSet = typename StringContainer::StringSet;
    using String = typename StringSet::String;
    tlx::LoserTreeCopy<false, String, dss_schimek::string_compare> lt(env.size());
    std::size_t filled_sources = 0;
    std::vector<String*> string_it;
    std::vector<String*> end_it;
    uint64_t curPos = 0;
    for (auto [start, size]: ranges) {
        string_it.push_back(stringContainer.strings() + curPos);
        end_it.push_back(stringContainer.strings() + curPos + size);
        curPos += size;
    }
    for (std::uint32_t i = 0; i < env.size(); ++i) {
        if (string_it[i] == end_it[i]) {
            lt.insert_start(nullptr, i, true);
        } else {
            lt.insert_start(&*string_it[i], i, false);
            ++filled_sources;
        }
    }
    lt.init();
    std::vector<String> result;
    result.reserve(stringContainer.size());
    while (filled_sources) {
        std::int32_t source = lt.min_source();
        result.push_back(*string_it[source]);
        ++string_it[source];
        if (string_it[source] != end_it[source]) {
            lt.delete_min_insert(&*string_it[source], false);
        } else {
            lt.delete_min_insert(nullptr, true);
            --filled_sources;
        }
    }
    stringContainer.set(std::move(result));
    return std::move(stringContainer);
}

} // namespace dss_schimek

namespace dss_mehnert {

template <
    typename StringPtr,
    typename PartitioningPolicy,
    typename SamplingPolicy,
    typename AllToAllStringPolicy>
class DistributedMergeSort : private PartitioningPolicy,
                             private SamplingPolicy,
                             private AllToAllStringPolicy {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    StringLcpContainer sort(
        StringPtr& local_string_ptr,
        StringLcpContainer&& local_string_container,
        dss_mehnert::Communicator comm = {}
    ) {
        using namespace dss_schimek;
        using namespace dss_schimek::measurement;

        MeasuringTool& measuring_tool = MeasuringTool::measuringTool();
        measuring_tool.setPhase("local_sorting");

        StringSet const& ss = local_string_ptr.active();
        const size_t lcpSummand = 5u;

        size_t charactersInSet = 0;
        for (auto const& str: ss) {
            charactersInSet += ss.get_length(str) + 1;
        }

        measuring_tool.add(charactersInSet, "charactersInSet");

        measuring_tool.start("sort_locally");
        if constexpr (AllToAllStringPolicy::Lcps) {
            tlx::sort_strings_detail::radixsort_CI3(local_string_ptr, 0, 0);
        } else {
            tlx::sort_strings_detail::radixsort_CI3(local_string_container.make_string_ptr(), 0, 0);
        }
        measuring_tool.stop("sort_locally");

        // There is only one PE, hence there is no need for distributed sorting
        if (comm.size() == 1) {
            return StringLcpContainer(std::move(local_string_container));
        }

        measuring_tool.setPhase("bucket_computation");
        measuring_tool.start("avg_lcp");
        auto globalLcpAvg = getAvgLcp(local_string_ptr) + lcpSummand;
        measuring_tool.stop("avg_lcp");

        // todo the dependant template is a bit ugly here
        std::vector<uint64_t> interval_sizes =
            PartitioningPolicy::template compute_partition<SamplingPolicy>(
                local_string_ptr,
                100 * globalLcpAvg,
                2,
                comm
            );

        measuring_tool.setPhase("string_exchange");
        comm.barrier();
        measuring_tool.start("all_to_all_strings");
        std::vector<std::size_t> receiving_interval_sizes =
            dss_schimek::mpi::alltoall(interval_sizes);
        StringLcpContainer recv_string_cont =
            AllToAllStringPolicy::alltoallv(local_string_container, interval_sizes);
        measuring_tool.stop("all_to_all_strings");

        measuring_tool.add(
            recv_string_cont.char_size() - recv_string_cont.size(),
            "num_received_chars",
            false
        );
        measuring_tool.add(recv_string_cont.size(), "num_recv_strings", false);
        measuring_tool.setPhase("merging");

        assert(num_recv_elems == recv_string_cont.size());

        measuring_tool.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(
                recv_string_cont,
                receiving_interval_sizes
            );
        measuring_tool.stop("compute_ranges");

        measuring_tool.start("merge_ranges");
        if constexpr (AllToAllStringPolicy::Lcps) {
            size_t num_recv_elems = std::accumulate(
                receiving_interval_sizes.begin(),
                receiving_interval_sizes.end(),
                static_cast<size_t>(0u)
            );
            auto sorted_container = choose_merge<AllToAllStringPolicy>(
                std::move(recv_string_cont),
                ranges,
                num_recv_elems
            );
            measuring_tool.stop("merge_ranges");
            measuring_tool.setPhase("none");
            return sorted_container;
        } else {
            auto sorted_container = noLcpMerge(std::move(recv_string_cont), ranges);
            measuring_tool.stop("merge_ranges");
            measuring_tool.setPhase("none");
            return sorted_container;
        }
    }
};

} // namespace dss_mehnert
