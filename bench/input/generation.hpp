// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Where the benchmark's strings come from. Takes its own Config rather than the parsed command
// line, so that input generation stays independent of how the benchmark is configured; the
// translation lives in SorterArgs::input_config.

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>

#include <kamping/collectives/barrier.hpp>
#include <kamping/measurements/counter.hpp>
#include <tlx/die/core.hpp>

#include "dss/mpi/communicator.hpp"
#include "dss/strings/stringcontainer.hpp"
#include "dss/util/measuringTool.hpp"
#include "input/string_generator.hpp"

namespace dss_mehnert {
namespace bench {
namespace input {

enum class StringGenerator {
    dn_ratio,
    dn_ratio_random,
    file,
    sentinel,
};

struct Config {
    StringGenerator generator = StringGenerator::dn_ratio;
    // the global number of strings to generate, already scaled for the number of PEs
    size_t num_strings = 0;
    size_t len_strings = 0;
    size_t len_strings_min = 0;
    size_t len_strings_max = 0;
    double dn_ratio = 0.5;
    bool dn_encode_padding = false;
    // skewed_dn_length: the fraction of the smallest strings whose length is drawn from an
    // interval that is skew_factor times longer, and which PE a string is generated on
    double skew_fraction = 0.0;
    double skew_factor = 1.0;
    // skewed_dn_length: pad the distinguishing prefix with a single constant character instead of
    // the tiled per-group encoding
    bool use_uniform_prefix = false;
    IdPlacement id_placement = IdPlacement::random;
    // skewed_dn_length: reproduce on a single PE the input a run with this many PEs would produce,
    // PE by PE in rank order (0 = generate normally)
    size_t simulate_num_pes = 0;
    size_t seed = 42;
    // for the file generator
    std::string path;
    // cap the number of bytes read from the file (0 = read the whole file)
    size_t max_num_bytes = 0;
};

inline void check_path_exists(std::string const& path) {
    tlx_die_verbose_unless(std::filesystem::exists(path), "file not found: " << path);
}

template <typename StringSet>
auto generate_strings(Config const& config, Communicator const& comm) {
    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_container = [&]() -> StringLcpContainer<StringSet> {
        switch (config.generator) {
            case StringGenerator::dn_ratio: {
                return DNRatioGenerator<StringSet>{
                    config.num_strings,
                    config.len_strings,
                    config.dn_ratio,
                    comm,
                    config.dn_encode_padding
                };
            }
            case StringGenerator::dn_ratio_random: {
                SkewedDNArgs const gen_args{
                    .global_strings = config.num_strings,
                    .min_length = config.len_strings_min,
                    .max_length = config.len_strings_max,
                    .use_uniform_prefix = config.use_uniform_prefix,
                    .dn_ratio = config.dn_ratio,
                    .skew_fraction = config.skew_fraction,
                    .skew_factor = config.skew_factor,
                    .placement = config.id_placement,
                    .seed = config.seed,
                };
                if (config.simulate_num_pes != 0) {
                    tlx_die_verbose_unless(
                        comm.size() == 1,
                        "--input-simulate-num-pes reproduces a distributed run on a "
                        "single PE, so "
                        "it has to be run with a single MPI rank"
                    );
                    return SkewedDNRatioLengthGenerator<StringSet>::simulate(
                        gen_args,
                        config.simulate_num_pes
                    );
                }
                return SkewedDNRatioLengthGenerator<StringSet>{gen_args, comm};
            }
            case StringGenerator::file: {
                check_path_exists(config.path);
                return FileDistributer<StringSet>{config.path, comm, config.max_num_bytes};
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

} // namespace input
} // namespace bench
} // namespace dss_mehnert
