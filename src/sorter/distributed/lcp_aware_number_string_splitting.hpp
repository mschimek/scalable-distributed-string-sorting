// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <random>
#include <type_traits>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/container/loser_tree.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/partition.hpp"
#include "tlx/die.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {

template <typename StringPtr, typename AllToAllStringPolicy, typename MergeSortImpl>
class DistributedMergeSort : protected AllToAllStringPolicy {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    // todo get rid of the option in
    std::enable_if_t<AllToAllStringPolicy::Lcps, StringLcpContainer>
    sort(StringPtr& string_ptr, StringLcpContainer&& container, Communicator const& comm) {
        using namespace dss_schimek;

        measuring_tool_.setPhase("local_sorting");

        StringSet const& ss = string_ptr.active();

        size_t charactersInSet = 0;
        for (auto const& str: ss) {
            charactersInSet += ss.get_length(str) + 1;
        }

        measuring_tool_.add(charactersInSet, "charactersInSet");

        measuring_tool_.start("sort_locally");
        tlx::sort_strings_detail::radixsort_CI3(string_ptr, 0, 0);
        measuring_tool_.stop("sort_locally");

        // There is only one PE, hence there is no need for distributed sorting
        if (comm.size() == 1) {
            return StringLcpContainer(std::move(container));
        }

        auto derived = static_cast<MergeSortImpl*>(this);
        return derived->impl(string_ptr, std::move(container), comm);
    }

protected:
    using MeasuringTool = dss_schimek::measurement::MeasuringTool;
    MeasuringTool measuring_tool_ = MeasuringTool::measuringTool();

    auto get_avg_lcp(StringPtr string_ptr) -> size_t {
        this->measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto avg_lcp = getAvgLcp(string_ptr) + lcp_summand;
        this->measuring_tool_.stop("avg_lcp");
        return avg_lcp;
    }
};


template <typename StringPtr, typename AllToAllStringPolicy, typename SamplerPolicy>
class SingleLevelMergeSort
    : public DistributedMergeSort<
          StringPtr,
          AllToAllStringPolicy,
          SingleLevelMergeSort<StringPtr, AllToAllStringPolicy, SamplerPolicy>> {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    static constexpr std::string_view getName() { return "SingleLevel"; }

    StringLcpContainer
    impl(StringPtr& string_ptr, StringLcpContainer&& container, Communicator const& comm) {
        using namespace dss_schimek;
        namespace kmp = kamping;

        this->measuring_tool_.setPhase("bucket_computation");
        auto global_lcp_avg = this->get_avg_lcp(string_ptr);

        constexpr auto compute_partition = partition::compute_partition<StringPtr, SamplerPolicy>;
        // todo why the *100 here
        // todo make sampling_factor variable
        auto interval_sizes =
            compute_partition(string_ptr, 100 * global_lcp_avg, comm.size(), 2, comm);

        this->measuring_tool_.setPhase("string_exchange");
        comm.barrier();

        this->measuring_tool_.start("all_to_all_strings");
        std::vector<std::size_t> receiving_interval_sizes =
            comm.alltoall(kmp::send_buf(interval_sizes)).exctract_recv_buffer();
        StringLcpContainer recv_string_cont =
            AllToAllStringPolicy::alltoallv(container, interval_sizes);
        this->measuring_tool_.stop("all_to_all_strings");

        this->measuring_tool_.add(
            recv_string_cont.char_size() - recv_string_cont.size(),
            "num_received_chars",
            false
        );
        this->measuring_tool_.add(recv_string_cont.size(), "num_recv_strings", false);
        this->measuring_tool_.setPhase("merging");

        assert(num_recv_elems == recv_string_cont.size());

        this->measuring_tool_.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(
                recv_string_cont,
                receiving_interval_sizes
            );
        this->measuring_tool_.stop("compute_ranges");

        this->measuring_tool_.start("merge_ranges");
        size_t num_recv_elems = std::accumulate(
            receiving_interval_sizes.begin(),
            receiving_interval_sizes.end(),
            static_cast<size_t>(0u)
        );
        auto sorted_container =
            choose_merge<AllToAllStringPolicy>(std::move(recv_string_cont), ranges, num_recv_elems);
        this->measuring_tool_.stop("merge_ranges");
        this->measuring_tool_.setPhase("none");
        return sorted_container;
    }
};


} // namespace dss_mehnert
