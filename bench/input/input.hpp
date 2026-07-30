// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Input generation and the small helpers the benchmarked algorithms share: the multi-level
// group sizes, the LCP/distinguishing-prefix statistics, and the MPI warmup.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <kamping/collectives/alltoall.hpp>
#include <kamping/collectives/barrier.hpp>
#include <kamping/measurements/counter.hpp>
#include <tlx/die/core.hpp>

#include "detail/args.hpp"
#include "dss/mpi/communicator.hpp"
#include "dss/mpi/is_sorted.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/util/measuringTool.hpp"
#include "input/string_generator.hpp"

namespace dss_mehnert {
namespace bench {

inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
}

inline auto get_first_level(std::vector<size_t> const& levels, Communicator const& comm) {
    return std::find_if(levels.begin(), levels.end(), [&](auto const& group_size) {
        return group_size < comm.size();
    });
}

// the input is partitioned once per level and once more in the final round
inline size_t get_num_levels(std::vector<size_t> const& levels, Communicator const& comm) {
    return static_cast<size_t>(std::distance(get_first_level(levels, comm), levels.end())) + 1;
}

template <typename StringSet>
auto generate_strings(SorterArgs const& args, Communicator const& comm) {
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_container = [&]() -> StringLcpContainer<StringSet> {
        switch (args.string_generator) {
            case StringGenerator::dn_ratio: {
                return DNRatioGenerator<StringSet>{
                    args.scaled_strings(comm),
                    args.len_strings,
                    args.dn_ratio,
                    comm,
                    args.dn_encode_padding
                };
            }
            case StringGenerator::dn_ratio_random: {
                SkewedDNArgs const gen_args{
                    .global_strings = args.scaled_strings(comm),
                    .min_length = args.len_strings_min,
                    .max_length = args.len_strings_max,
                    .use_uniform_prefix = args.use_uniform_prefix,
                    .dn_ratio = args.dn_ratio,
                    .skew_fraction = args.skew_fraction,
                    .skew_factor = args.skew_factor,
                    .placement = args.id_placement,
                    .seed = args.seed,
                };
                if (args.simulate_num_pes != 0) {
                    tlx_die_verbose_unless(
                        comm.size() == 1,
                        "--input-simulate-num-pes reproduces a distributed run on a "
                        "single PE, so "
                        "it has to be run with a single MPI rank"
                    );
                    return SkewedDNRatioLengthGenerator<StringSet>::simulate(
                        gen_args,
                        args.simulate_num_pes
                    );
                }
                return SkewedDNRatioLengthGenerator<StringSet>{gen_args, comm};
            }
            case StringGenerator::file: {
                check_path_exists(args.path);
                return FileDistributer<StringSet>{args.path, comm, args.max_num_bytes};
            }

            case StringGenerator::sentinel: {
                break;
            }
        };
        tlx_die("invalid string generator");
    }();
    measuring_tool.stop("generate_strings");

    comm.barrier();

    auto const num_gen_chars = input_container.char_size();
    auto const num_gen_strs = input_container.size();
    measuring_tool.add(num_gen_chars - num_gen_strs, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");

    using kamping::measurements::GlobalAggregationMode;
    std::vector<GlobalAggregationMode> const agg{
        GlobalAggregationMode::min,
        GlobalAggregationMode::max,
        GlobalAggregationMode::sum,
    };
    kamping::measurements::counter()
        .add("input_chars", static_cast<std::int64_t>(num_gen_chars - num_gen_strs), agg);
    kamping::measurements::counter()
        .add("input_strings", static_cast<std::int64_t>(num_gen_strs), agg);

    return input_container;
}

template <typename Container>
void count_prefix_lengths(Container& container, Communicator const& comm) {
    using namespace kamping;

    using Char = Container::Char;
    std::vector<Char> const last_string =
        container.empty() ? std::vector<Char>{0} : container.get_raw_string(container.size() - 1);
    auto const pred_string = get_predecessor(last_string, container.empty(), comm);

    size_t first_lcp = 0;
    if (pred_string && !container.empty()) {
        auto const first_string = container.get_raw_string(0);
        first_lcp = dss_schimek::calc_lcp(pred_string->data(), first_string.data());
    }

    size_t const last_lcp = container.empty() ? 0 : container.lcps().back();
    auto const pred_lcp =
        container.size() == 1 ? first_lcp : get_predecessor(last_lcp, container.empty(), comm);

    auto const begin = container.lcps().begin(), end = container.lcps().end();
    auto const local_lcp = first_lcp + std::accumulate(begin, end, size_t{0});

    auto const dist = [](auto const lcp1, auto const lcp2) { return std::max(lcp1, lcp2) + 1; };

    auto local_dist = container.empty() || !pred_lcp ? 0 : dist(*pred_lcp, first_lcp);
    if (!container.empty()) {
        local_dist =
            std::transform_reduce(std::next(begin), end, begin, size_t{0}, std::plus<>{}, dist);
    }

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    measuring_tool.add(local_lcp, "global_lcp_sum");
    measuring_tool.add(local_dist, "global_dist_prefix");
}

// Every PE sends a different, random number of bytes to every other PE, drawn uniformly from
// [min_bytes, max_bytes).
inline size_t
mpi_irregular_warmup(size_t const min_bytes, size_t const max_bytes, Communicator const& comm) {
    tlx_die_unless(min_bytes < max_bytes);

    std::mt19937_64 gen{comm.rank()};
    // uniform_int_distribution is inclusive on both ends, so draw from [min_bytes, max_bytes - 1]
    std::uniform_int_distribution<size_t> count_dist{min_bytes, max_bytes - 1};
    std::uniform_int_distribution<unsigned char> byte_dist{'A', 'Z'};

    std::vector<int> send_counts(comm.size());
    std::generate(send_counts.begin(), send_counts.end(), [&] {
        return static_cast<int>(count_dist(gen));
    });

    size_t const send_total = std::accumulate(send_counts.begin(), send_counts.end(), size_t{0});
    std::vector<unsigned char> send_data(send_total);
    std::generate(send_data.begin(), send_data.end(), [&] { return byte_dist(gen); });

    // recv_counts are exchanged internally by kamping from the send_counts
    auto recv_data =
        comm.alltoallv(kamping::send_buf(send_data), kamping::send_counts(send_counts));

    auto volatile sum = std::accumulate(recv_data.begin(), recv_data.end(), size_t{0});
    return sum;
}

} // namespace bench
} // namespace dss_mehnert
