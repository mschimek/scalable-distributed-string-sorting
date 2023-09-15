// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <kamping/collectives/allgather.hpp>

#include "mpi/communicator.hpp"
#include "sorter/RQuick/RQuick.hpp"
#include "sorter/RQuick2/RQuick.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/sample.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace partition {

using dss_mehnert::sample::SampleParams;

template <typename Char, bool is_indexed>
using SorterStringSet = std::conditional_t<
    is_indexed,
    dss_schimek::GenericCharLengthIndexStringSet<Char>,
    dss_schimek::GenericCharLengthStringSet<Char>>;

template <bool is_indexed, typename Derived>
class PartitionPolicy {
public:
    template <typename StringPtr, typename Sample>
    static std::vector<size_t> compute_partition(
        StringPtr const& strptr,
        Sample&& sample,
        size_t const num_partitions,
        Communicator const& comm
    ) {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        measuring_tool.start("sort_samples");
        auto sorted_sample = Derived::sort_samples(std::move(sample), comm);
        measuring_tool.stop("sort_samples");

        measuring_tool.start("choose_splitters");
        auto sample_set = sorted_sample.make_string_set();
        auto chosen_splitters = Derived::choose_splitters(sample_set, num_partitions, comm);
        measuring_tool.stop("choose_splitters");

        measuring_tool.start("compute_interval_sizes");
        auto splitter_set = chosen_splitters.make_string_set();
        std::vector<size_t> interval_sizes;
        if constexpr (is_indexed) {
            // tood this is being computed redundantly multiple times
            auto local_offset = sample::get_local_offset(strptr.size(), comm);
            interval_sizes =
                compute_interval_binary_index(strptr.active(), splitter_set, local_offset);
        } else {
            interval_sizes = compute_interval_binary(strptr.active(), splitter_set);
        }
        measuring_tool.stop("compute_interval_sizes");

        return interval_sizes;
    }
};

template <typename Char, bool is_indexed>
class RQuickV1 : public PartitionPolicy<is_indexed, RQuickV1<Char, is_indexed>> {
    friend PartitionPolicy<is_indexed, RQuickV1<Char, is_indexed>>;

private:
    static constexpr int tag = 50352;
    static constexpr uint64_t seed = 3469931;
    static constexpr bool is_robust = true;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringPtr = tlx::sort_strings_detail::StringPtr<StringSet>;

    template <typename Samples>
    static dss_schimek::StringContainer<StringSet>
    sort_samples(Samples&& samples, Communicator const& comm) {
        RQuick2::Comparator<StringPtr> const comp;
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        RQuick::Data<dss_schimek::StringContainer<StringSet>, is_indexed> data;
        if constexpr (is_indexed) {
            data = {std::move(samples.sample), {samples.indices}};
        } else {
            data = {std::move(samples), {}};
        }
        return RQuick::sort(gen, std::move(data), MPI_BYTE, tag, comm_mpi, comp, is_robust);
    }

    static dss_schimek::StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const& comm) {
        if constexpr (is_indexed) {
            return distributed_choose_splitters_indexed(ss, num_partitions, comm);
        } else {
            return distributed_choose_splitters(ss, num_partitions, comm);
        }
    }
};

template <typename Char, bool is_indexed, bool use_lcps>
class RQuickV2 : public PartitionPolicy<is_indexed, RQuickV2<Char, is_indexed, use_lcps>> {
    friend PartitionPolicy<is_indexed, RQuickV2<Char, is_indexed, use_lcps>>;

private:
    static constexpr int tag = 23560;
    static constexpr uint64_t seed = 3469931;

    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringPtr = std::conditional_t<
        use_lcps,
        tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>,
        tlx::sort_strings_detail::StringPtr<StringSet>>;

    template <typename Samples>
    static RQuick2::Container<StringPtr> sort_samples(Samples&& samples, Communicator const& comm) {
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        // todo not a huge fan of this
        RQuick2::Data<StringPtr> data;
        if constexpr (is_indexed) {
            data.raw_strs = std::move(samples.sample);
            data.indices = std::move(samples.indices);
        } else {
            data.raw_strs = std::move(samples);
        }
        // LCP array initialization is done by RQuick
        return RQuick2::sort(std::move(data), tag, gen, comm_mpi);
    }

    static dss_schimek::StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const& comm) {
        if constexpr (is_indexed) {
            return distributed_choose_splitters_indexed(ss, num_partitions, comm);
        } else {
            return distributed_choose_splitters(ss, num_partitions, comm);
        }
    }
};

// todo consider adding a CLI option for this
template <typename Char, bool is_indexed>
class Sequential : public PartitionPolicy<is_indexed, Sequential<Char, is_indexed>> {
    friend PartitionPolicy<is_indexed, Sequential<Char, is_indexed>>;

private:
    using StringSet = SorterStringSet<Char, is_indexed>;
    using StringContainer = dss_schimek::StringLcpContainer<StringSet>;

    template <typename Samples>
    static StringContainer sort_samples(Samples&& samples, Communicator const& comm) {
        StringContainer global_samples;
        if constexpr (is_indexed) {
            auto recv_sample = comm.allgatherv(kamping::send_buf(samples.sample));
            auto recv_indices = comm.allgatherv(kamping::send_buf(samples.indices));
            // todo this constructor is not implemented right now
            global_samples = StringContainer{
                recv_sample.extract_recv_buffer(),
                recv_indices.extract_recv_buffer()};
        } else {
            auto recv_sample = comm.allgatherv(kamping::send_buf(samples));
            global_samples = StringContainer{recv_sample.extract_recv_buffer()};
        }

        tlx::sort_strings_detail::radixsort_CI3(global_samples.make_string_lcp_ptr(), 0, 0);
        if constexpr (is_indexed) {
            sort_duplicates(global_samples.make_string_lcp_ptr());
        }

        return global_samples;
    }

    static dss_schimek::StringContainer<StringSet>
    choose_splitters(StringSet const& ss, size_t const num_partitions, Communicator const&) {
        return choose_splitters(ss, num_partitions);
    }
};

} // namespace partition
} // namespace dss_mehnert
