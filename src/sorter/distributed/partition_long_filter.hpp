// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <kamping/collectives/allgather.hpp>
#include <kamping/collectives/allreduce.hpp>
#include <kamping/collectives/exscan.hpp>
#include <kamping/measurements/counter.hpp>
#include <kamping/measurements/timer.hpp>
#include <kamping/mpi_ops.hpp>
#include <kamping/named_parameters.hpp>

#include "mpi/communicator.hpp"
#include "sorter/RQuick2/RQuick.hpp"
#include "sorter/RQuick2/Util.hpp"
#include "sorter/distributed/misc.hpp"
#include "sorter/distributed/partition.hpp"
#include "sorter/distributed/sample.hpp"
#include "sorter/distributed/sample_redistribution.hpp"
#include "strings/stringcontainer.hpp"
#include "util/measuringTool.hpp"

namespace dss_mehnert {
namespace partition {

// Splitter policy that filters "long" strings out of the splitter sample before
// sorting it (see PLAN_long_splitter_filter.md). Char-based sampling can place a
// few very long strings among many short ones; because RQuick balances the
// *number* of strings per PE rather than characters, sorting such a sample is
// badly imbalanced. This policy:
//   1. computes a length threshold l_thr = max(1, alpha * avg_sample_len),
//   2. splits the sample into shorts (len < l_thr) and longs (len >= l_thr),
//   3. globally sorts the longs (round 1, distributed RQuick2) so each long gets
//      a global *rank*,
//   4. runs the stock indexed RQuick2 (round 2) over the shorts (indexed by
//      input id) together with the longs truncated to l_thr chars (indexed by
//      rank, overloading the Index slot),
//   5. after choosing splitters, de-truncates the chosen long splitters back to
//      their full characters and original input id.
//
// The combined round-2 order equals the true lexicographic order of the
// original sample strings (tie-broken by input id), so the chosen splitters --
// and hence the interval sizes -- are identical to plain indexed RQuickV2.
//
// Unlike the other policies it overrides select_splitters wholesale (rather than
// the static sort_samples/choose_splitters hooks of BaseSplitterPolicy) because
// the sorted long container must survive from round 1 until de-truncation.
// PartitionPolicy::compute_partition calls SplitterPolicy::select_splitters
// unqualified, so this override is picked up without touching the other policies.
//
// Round 1 sorts the longs distributively, so after it no PE holds the whole
// sorted long list. De-truncation therefore fetches the full chars + input id of
// just the chosen long splitters (<= p-1 of them): every PE contributes the
// chosen longs whose global rank falls in its local block, and one allgatherv
// unions them onto every PE. This never materializes the full long set on a
// single PE.
template <typename Char, bool is_indexed, bool use_lcps>
class RQuickV2LongFilter {
    static_assert(is_indexed, "RQuickV2LongFilter currently supports indexed mode only");

public:
    using Sample = sample::SampleResult<Char, is_indexed>;

    RQuickV2LongFilter() = default;

    explicit RQuickV2LongFilter(double const long_threshold_factor)
        : long_threshold_factor_{long_threshold_factor} {}

    template <typename StringPtr>
    StringContainer<SorterStringSet<Char, true>> select_splitters(
        StringPtr const&,
        Sample&& sample,
        size_t const num_partitions,
        Communicator const& comm,
        bool const redistribute_sample
    ) const {
        auto& measuring_tool = measurement::MeasuringTool::measuringTool();

        // count sample size before it is consumed by the sort (mirrors
        // BaseSplitterPolicy so the counters stay comparable across policies)
        using kamping::measurements::GlobalAggregationMode;
        std::vector<GlobalAggregationMode> const agg{
            GlobalAggregationMode::min,
            GlobalAggregationMode::max,
            GlobalAggregationMode::sum,
        };
        auto const num_sample_strings = static_cast<std::int64_t>(
            std::count(sample.sample.begin(), sample.sample.end(), Char{0})
        );
        auto const num_sample_chars =
            static_cast<std::int64_t>(sample.sample.size()) - num_sample_strings;
        kamping::measurements::counter().add("sample_num_strings", num_sample_strings, agg);
        kamping::measurements::counter().add("sample_num_chars", num_sample_chars, agg);

        // threshold l_thr = max(1, alpha * avg_sample_len)
        auto const totals = comm.allreduce(
            kamping::send_buf({num_sample_chars, num_sample_strings}),
            kamping::op(kamping::ops::plus<>{})
        );
        auto const total_chars = static_cast<double>(totals[0]);
        auto const total_strings = static_cast<double>(totals[1]);
        double const avg_len = total_strings == 0.0 ? 0.0 : total_chars / total_strings;
        size_t const l_thr =
            std::max<size_t>(1, static_cast<size_t>(long_threshold_factor_ * avg_len));

        measuring_tool.start("sort_samples");
        comm.barrier();
        kamping::measurements::timer().start("sort_samples");

        // prior to any sorting, pseudorandomly permute the sample across the PEs
        // so RQuick's per-string-count balancing starts from a balanced sample
        // (a PE that sampled a few very long strings would otherwise stay skewed).
        // qualified because the 'sample' parameter shadows the sample namespace.
        if (redistribute_sample) {
            sample = dss_mehnert::sample::redistribute_random_timed(std::move(sample), comm);
        }

        // pass 1: count local long strings to decide on the fast path
        size_t local_long_count = 0;
        for (auto it = sample.sample.begin(); it != sample.sample.end();) {
            auto const str_begin = it;
            for (; *it != Char{0}; ++it) {}
            if (static_cast<size_t>(it - str_begin) >= l_thr) {
                ++local_long_count;
            }
            ++it; // skip terminator
        }
        size_t const global_long_count = comm.allreduce_single(
            kamping::send_buf(local_long_count),
            kamping::op(kamping::ops::plus<>{})
        );

        // fast path: no long strings anywhere -> stock indexed RQuick2 on the
        // untouched sample (the split/round-1 machinery would be a no-op)
        if (global_long_count == 0) {
            kamping::measurements::timer().start("sort_short_samples");
            auto sorted = sort_indexed(std::move(sample), comm, tag_round2);
            kamping::measurements::timer().stop_and_append();
            kamping::measurements::timer().stop_and_append();
            measuring_tool.stop("sort_samples");

            measuring_tool.start("choose_splitters");
            kamping::measurements::timer().start("choose_splitters");
            auto sample_set = sorted.make_string_set();
            auto chosen = choose_splitters_distributed(sample_set, num_partitions, comm);
            kamping::measurements::timer().stop_and_append();
            measuring_tool.stop("choose_splitters");
            return chosen;
        }

        // pass 2: split the flat sample into shorts and longs
        std::vector<Char> short_chars, long_chars;
        std::vector<uint64_t> short_indices, long_indices;
        short_chars.reserve(sample.sample.size());
        short_indices.reserve(sample.indices.size());
        {
            auto it = sample.sample.begin();
            for (size_t i = 0; it != sample.sample.end(); ++i) {
                auto const str_begin = it;
                for (; *it != Char{0}; ++it) {}
                auto const str_end = it;
                ++it; // skip terminator

                size_t const len = static_cast<size_t>(str_end - str_begin);
                uint64_t const input_id = sample.indices[i];
                if (len < l_thr) {
                    short_chars.insert(short_chars.end(), str_begin, str_end);
                    short_chars.push_back(Char{0});
                    short_indices.push_back(input_id);
                } else {
                    long_chars.insert(long_chars.end(), str_begin, str_end);
                    long_chars.push_back(Char{0});
                    long_indices.push_back(input_id);
                }
            }
        }

        // round 1: globally sort the long strings with distributed RQuick2,
        // indexed by input id. Each PE keeps a contiguous block of the globally
        // sorted longs; the global rank of a local long is its exclusive-prefix
        // offset plus its local position. Equal longs are ordered by input id
        // (RQuick2 sorts duplicates by index), so rank order == (string, id)
        // order -- exactly what round 2 needs to complete l_thr-prefix ties.
        kamping::measurements::timer().start("sort_long_samples");
        auto sorted_longs =
            sort_indexed(make_sample(std::move(long_chars), std::move(long_indices)), comm, tag_round1);
        kamping::measurements::timer().stop_and_append();
        size_t const rank_offset = comm.exscan_single(
            kamping::send_buf(sorted_longs.size()),
            kamping::op(kamping::ops::plus<>{})
        );
        auto const long_ss = sorted_longs.make_string_set();

        // round 2: shorts (indexed by input id) + this PE's truncated longs
        // (indexed by their global rank). The union over PEs of the truncated
        // longs is the full long set, so the global round-2 multiset equals the
        // original sample with the longs truncated to l_thr chars.
        kamping::measurements::timer().start("truncate_longs");
        std::vector<Char> r2_chars = std::move(short_chars);
        std::vector<uint64_t> r2_indices = std::move(short_indices);
        r2_chars.reserve(r2_chars.size() + sorted_longs.size() * (l_thr + 1));
        r2_indices.reserve(r2_indices.size() + sorted_longs.size());
        for (size_t j = 0; j != sorted_longs.size(); ++j) {
            auto const chars = long_ss.get_chars(sorted_longs[j], 0);
            r2_chars.insert(r2_chars.end(), chars, chars + l_thr);
            r2_chars.push_back(Char{0});
            r2_indices.push_back(rank_offset + j); // global rank in the Index slot
        }
        kamping::measurements::timer().stop_and_append();

        kamping::measurements::timer().start("sort_short_samples");
        auto sorted_r2 = sort_indexed(
            make_sample(std::move(r2_chars), std::move(r2_indices), sample.local_offset),
            comm,
            tag_round2
        );
        kamping::measurements::timer().stop_and_append();

        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("sort_samples");

        measuring_tool.start("choose_splitters");
        kamping::measurements::timer().start("choose_splitters");
        auto r2_set = sorted_r2.make_string_set();
        auto chosen = choose_splitters_distributed(r2_set, num_partitions, comm);

        // de-truncate: chosen splitters of length exactly l_thr are truncated
        // longs whose Index slot holds a global rank; replace them with the full
        // chars + original input id (fetched via allgatherv, see class comment).
        kamping::measurements::timer().start("detruncate_splitters");
        auto result = detruncate_splitters(chosen, sorted_longs, rank_offset, l_thr, comm);
        kamping::measurements::timer().stop_and_append();

        kamping::measurements::timer().stop_and_append();
        measuring_tool.stop("choose_splitters");

        return result;
    }

private:
    static constexpr int tag_round1 = 29017;
    static constexpr int tag_round2 = 29117;
    static constexpr uint64_t seed = 3469931;

    double long_threshold_factor_ = 5.0;

    using IndexedStringSet = SorterStringSet<Char, true>;
    using R2StringPtr = std::conditional_t<
        use_lcps,
        tlx::sort_strings_detail::StringLcpPtr<IndexedStringSet, size_t>,
        tlx::sort_strings_detail::StringPtr<IndexedStringSet>>;
    using SortedContainer = RQuick2::Container<R2StringPtr>;

    // a gathered long splitter, referencing chars in the allgatherv receive buffer
    struct LongRef {
        size_t begin;
        size_t length;
        uint64_t input_id;
    };

    static Sample
    make_sample(std::vector<Char>&& chars, std::vector<uint64_t>&& indices, size_t const local_offset = 0) {
        Sample sample;
        sample.sample = std::move(chars);
        sample.indices = std::move(indices);
        sample.local_offset = local_offset;
        return sample;
    }

    // indexed RQuick2 sort, matching RQuickV2::sort_samples
    static SortedContainer sort_indexed(Sample&& sample, Communicator const& comm, int const tag) {
        std::mt19937_64 gen{seed + comm.rank()};
        auto const comm_mpi = comm.mpi_communicator();

        RQuick2::Data<R2StringPtr> data{std::move(sample.sample)};
        data.indices = std::move(sample.indices);
        // LCP array initialization is done by RQuick
        return RQuick2::sort(std::move(data), tag, gen, comm_mpi);
    }

    // replace truncated-long splitters with their full chars + input id. The
    // sorted longs are distributed, so each PE contributes the chosen longs it
    // owns (rank in [rank_offset, rank_offset + local size)) and one allgatherv
    // unions them; every chosen long is owned by exactly one PE.
    static StringContainer<IndexedStringSet> detruncate_splitters(
        StringContainer<IndexedStringSet>& chosen,
        SortedContainer& sorted_longs,
        size_t const rank_offset,
        size_t const l_thr,
        Communicator const& comm
    ) {
        auto const chosen_ss = chosen.make_string_set();
        auto const long_ss = sorted_longs.make_string_set();
        size_t const local_count = sorted_longs.size();

        // gather the full data of every chosen long this PE owns
        std::vector<Char> send_chars;
        std::vector<uint64_t> send_ranks, send_ids;
        for (auto const& splitter: chosen.get_strings()) {
            if (chosen_ss.get_length(splitter) != l_thr) {
                continue; // short splitter -- already complete
            }
            uint64_t const rank = splitter.index;
            if (rank < rank_offset || rank >= rank_offset + local_count) {
                continue; // owned by another PE
            }
            auto const& full = sorted_longs[rank - rank_offset];
            auto const chars = long_ss.get_chars(full, 0);
            send_chars.insert(send_chars.end(), chars, chars + long_ss.get_length(full));
            send_chars.push_back(Char{0});
            send_ranks.push_back(rank);
            send_ids.push_back(full.index);
        }

        auto recv_chars = comm.allgatherv(kamping::send_buf(send_chars));
        auto recv_ranks = comm.allgatherv(kamping::send_buf(send_ranks));
        auto recv_ids = comm.allgatherv(kamping::send_buf(send_ids));

        // index the gathered longs by rank
        std::unordered_map<uint64_t, LongRef> long_by_rank;
        long_by_rank.reserve(recv_ranks.size());
        for (size_t i = 0, pos = 0; i != recv_ranks.size(); ++i) {
            size_t const begin = pos;
            while (recv_chars[pos] != Char{0}) {
                ++pos;
            }
            long_by_rank.emplace(recv_ranks[i], LongRef{begin, pos - begin, recv_ids[i]});
            ++pos; // skip terminator
        }

        // rebuild the splitters in order, de-truncating the longs
        std::vector<Char> final_chars;
        std::vector<uint64_t> final_indices;
        final_chars.reserve(chosen.char_size());
        final_indices.reserve(chosen.size());
        for (auto const& splitter: chosen.get_strings()) {
            size_t const len = chosen_ss.get_length(splitter);
            if (len == l_thr) {
                auto const& ref = long_by_rank.at(splitter.index);
                auto const begin = recv_chars.data() + ref.begin;
                final_chars.insert(final_chars.end(), begin, begin + ref.length);
                final_indices.push_back(ref.input_id);
            } else {
                auto const chars = chosen_ss.get_chars(splitter, 0);
                final_chars.insert(final_chars.end(), chars, chars + len);
                final_indices.push_back(splitter.index);
            }
            final_chars.push_back(Char{0});
        }

        return StringContainer<IndexedStringSet>{
            std::move(final_chars),
            make_initializer<Index>(final_indices)};
    }
};

} // namespace partition
} // namespace dss_mehnert
