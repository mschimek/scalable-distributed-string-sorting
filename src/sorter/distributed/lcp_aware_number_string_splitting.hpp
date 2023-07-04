// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <iterator>
#include <random>
#include <string_view>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/named_parameters.hpp>
#include <tlx/container/loser_tree.hpp>
#include <tlx/die.hpp>
#include <tlx/sort/strings/radix_sort.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "mpi/communicator.hpp"
#include "sorter/distributed/merging.hpp"
#include "sorter/distributed/partition.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {

// todo investigate wheter RQuick is working correctly (in particular power oft two issue)
template <typename StringPtr, typename AllToAllStringPolicy, typename MergeSortImpl>
class DistributedMergeSort : protected AllToAllStringPolicy {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    // todo get rid of Lcps in AllToAll
    StringLcpContainer
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
    MeasuringTool& measuring_tool_ = MeasuringTool::measuringTool();

    size_t get_avg_lcp(StringPtr string_ptr, Communicator const& comm) {
        this->measuring_tool_.start("avg_lcp");
        const size_t lcp_summand = 5u;
        auto avg_lcp = getAvgLcp(string_ptr, comm) + lcp_summand;
        this->measuring_tool_.stop("avg_lcp");
        return avg_lcp;
    }

    std::pair<StringLcpContainer, std::vector<size_t>> string_exchange(
        StringLcpContainer&& container,
        std::vector<size_t> const& interval_sizes,
        Communicator const& comm
    ) {
        this->measuring_tool_.start("all_to_all_strings");
        std::vector<size_t> recv_interval_sizes =
            comm.alltoall(kamping::send_buf(interval_sizes)).extract_recv_buffer();
        StringLcpContainer recv_string_container =
            AllToAllStringPolicy::alltoallv(container, interval_sizes, comm);
        this->measuring_tool_.stop("all_to_all_strings");

        auto num_received_chars = recv_string_container.char_size() - recv_string_container.size();
        this->measuring_tool_.add(num_received_chars, "num_received_chars", false);
        this->measuring_tool_.add(recv_string_container.size(), "num_recv_strings", false);

        return {std::move(recv_string_container), std::move(recv_interval_sizes)};
    }

    StringLcpContainer merge_ranges(
        StringLcpContainer&& container,
        std::vector<size_t>& interval_sizes,
        Communicator const& comm
    ) {
        this->measuring_tool_.start("compute_ranges");
        std::vector<std::pair<size_t, size_t>> ranges =
            compute_ranges_and_set_lcp_at_start_of_range(container, interval_sizes, comm);
        this->measuring_tool_.stop("compute_ranges");

        this->measuring_tool_.start("merge_ranges");
        size_t num_recv_elems =
            std::accumulate(interval_sizes.begin(), interval_sizes.end(), static_cast<size_t>(0u));
        assert(num_recv_elems == recv_string_cont.size());

        auto sorted_container =
            choose_merge<AllToAllStringPolicy>(std::move(container), ranges, num_recv_elems, comm);
        this->measuring_tool_.stop("merge_ranges");

        return sorted_container;
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

        this->measuring_tool_.setPhase("bucket_computation");
        auto global_lcp_avg = this->get_avg_lcp(string_ptr, comm);

        constexpr auto compute_partition = partition::compute_partition<StringPtr, SamplerPolicy>;
        // todo why the *100 here
        // todo make sampling_factor variable
        auto interval_sizes =
            compute_partition(string_ptr, 100 * global_lcp_avg, comm.size(), 2, comm);

        comm.barrier();
        this->measuring_tool_.setPhase("string_exchange");
        auto [recv_string_container, recv_interval_sizes] =
            this->string_exchange(std::move(container), interval_sizes, comm);

        this->measuring_tool_.setPhase("merging");
        auto sorted_container =
            this->merge_ranges(std::move(recv_string_container), recv_interval_sizes, comm);

        this->measuring_tool_.setPhase("none");
        return sorted_container;
    }
};

template <typename StringPtr, typename AllToAllStringPolicy, typename SamplerPolicy>
class MultiLevelMergeSort
    : public DistributedMergeSort<
          StringPtr,
          AllToAllStringPolicy,
          MultiLevelMergeSort<StringPtr, AllToAllStringPolicy, SamplerPolicy>> {
public:
    using StringSet = typename StringPtr::StringSet;
    using StringLcpContainer = dss_schimek::StringLcpContainer<StringSet>;

    static constexpr std::string_view getName() { return "MultiLevel"; }

    StringLcpContainer
    impl(StringPtr& string_ptr, StringLcpContainer&& container, Communicator const& comm) {
        using namespace dss_schimek;

        this->measuring_tool_.setPhase("bucket_computation");
        auto global_lcp_avg = this->get_avg_lcp(string_ptr);

        size_t group_size{3};                        // todo make this variable
        size_t num_groups{comm.size() / group_size}; // todo consider rounding
        die_unless(comm.size() % group_size == 0);   // todo allow this

        constexpr auto compute_partition = partition::compute_partition<StringPtr, SamplerPolicy>;
        // todo why the *100 here
        // todo make sampling_factor variable
        auto interval_sizes =
            compute_partition(string_ptr, 100 * global_lcp_avg, num_groups, 2, comm);

        size_t group_offset{comm.rank() / num_groups};
        std::vector<uint64_t> send_counts(comm.size());

        // todo replace with non-naive strategy
        auto dst_iter = std::begin(send_counts) + group_offset;
        for (auto const interval: interval_sizes) {
            *dst_iter = interval;
            std::advance(dst_iter, group_size);
        }

        comm.barrier();
        this->measuring_tool_.setPhase("string_exchange");
        auto [recv_string_container, recv_interval_sizes] =
            this->string_exchange(std::move(container), send_counts, comm);

        this->measuring_tool_.setPhase("merging");
        auto sorted_container =
            this->merge_ranges(std::move(recv_string_container), recv_interval_sizes, comm);

        this->measuring_tool_.setPhase("none");
        return sorted_container;
    }
};

} // namespace dss_mehnert
