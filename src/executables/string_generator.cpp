// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Writes the input of a skewedDNLenGen (string generator 5) run to a file, on a single PE.
//
// The point is not to generate *an* input with the same parameters -- distributed_sorter can do
// that on its own -- but the input of one particular run: the strings of PE 0, then those of PE 1,
// and so on, byte for byte as those PEs would hold them. SkewedDNRatioLengthGenerator::simulate
// replays every PE's random draws in order to reproduce that on one PE, so the file can be handed
// to a shared memory sorter (or any other tool) as the very same instance a distributed run sorted.
//
// The arguments mirror distributed_sorter's, so the command line of the run being reproduced
// carries over unchanged apart from --num-pes and --output.
//
// The strings are written one per line. The generator's alphabet is 'A'-'Z', so no string can
// contain a newline and the file round-trips through FileDistributer (--string-generator 2).

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/spdlog.h>
#include <tlx/die/core.hpp>

#include "mpi/communicator.hpp"
#include "strings/stringset.hpp"
#include "util/string_generator.hpp"

namespace {

using CharType = unsigned char;
using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
using Generator = dss_mehnert::SkewedDNRatioLengthGenerator<StringSet>;

constexpr CharType newline = '\n';

struct GeneratorArgs {
    // the run being reproduced: how many PEs it had, and how many strings each of them was asked
    // for. The global count is derived exactly as distributed_sorter derives it, so that passing
    // that run's --num-strings and --strong-scaling here yields the same instance.
    size_t num_pes = 0;
    size_t num_strings = 100000;
    bool strong_scaling = false;

    size_t len_strings_min = 100;
    size_t len_strings_max = 110;
    double dn_ratio = 0.5;
    double skew_fraction = 0.0;
    double skew_factor = 1.0;
    bool use_uniform_prefix = false;
    size_t id_placement = static_cast<size_t>(dss_mehnert::IdPlacement::random);
    size_t seed = 42;

    std::string output_path;

    size_t global_strings() const { return (strong_scaling ? 1 : num_pes) * num_strings; }
};

void add_generator_args(GeneratorArgs& args, CLI::App& app) {
    app.add_option(
           "--num-pes",
           args.num_pes,
           "number of PEs of the run to reproduce; the file holds their strings concatenated in "
           "rank order"
    )
        ->required()
        ->group("Input");
    app.add_option(
           "--num-strings",
           args.num_strings,
           "number of strings per PE, or in total with --strong-scaling"
    )
        ->group("Input");
    app.add_flag(
           "--strong-scaling",
           args.strong_scaling,
           "interpret --num-strings as the global count instead of a per-PE one"
    )
        ->group("Input");
    app.add_option("--min-len-strings", args.len_strings_min, "minimum length of generated strings")
        ->group("Input");
    app.add_option("--max-len-strings", args.len_strings_max, "maximum length of generated strings")
        ->group("Input");
    app.add_option("--DN-ratio", args.dn_ratio, "D/N ratio of generated strings")->group("Input");
    app.add_option(
           "--skew-fraction",
           args.skew_fraction,
           "the fraction of the smallest strings that are stretched; their length is drawn from "
           "[min-len-strings, skew-factor * max-len-strings]"
    )
        ->group("Input");
    app.add_option(
           "--skew-factor",
           args.skew_factor,
           "the factor by which the stretched strings may be longer"
    )
        ->group("Input");
    app.add_flag(
           "--input-use-uniform-prefix",
           args.use_uniform_prefix,
           "pad the distinguishing prefix with a single constant character instead of the tiled "
           "per-group encoding"
    )
        ->group("Input");
    app.add_option(
           "--placement",
           args.id_placement,
           "which PE a string is generated on ([0]=random, 1=contiguous)"
    )
        ->group("Input");
    app.add_option("--seed", args.seed, "base seed for input generation (default 42)")
        ->group("Input");

    app.add_option("--output", args.output_path, "path of the file to write the strings to")
        ->required()
        ->group("Output");
}

// the clamping common_cli.hpp applies to enum-valued options, without pulling the sorter dispatch
// tree into this executable
dss_mehnert::IdPlacement clamp_placement(size_t const value) {
    return static_cast<dss_mehnert::IdPlacement>(
        std::min(value, static_cast<size_t>(dss_mehnert::IdPlacement::sentinel))
    );
}

// The container holds the strings as `str \0 str \0 ...` in exactly the order the PEs would hold
// them, so turning the terminators into newlines is the whole conversion. Done in place, because
// the buffer is the size of the input and a second copy of it is the thing this tool is trying to
// avoid needing.
size_t to_lines(std::vector<CharType>& raw_strings) {
    size_t num_lines = 0;
    for (auto& c: raw_strings) {
        if (c == CharType{0}) {
            c = newline;
            ++num_lines;
        }
    }
    return num_lines;
}

} // namespace

int main(int argc, char* argv[]) {
    GeneratorArgs args;

    CLI::App app{"writes the input of a skewedDNLenGen run to a file"};
    app.option_defaults()->always_capture_default();
    add_generator_args(args, app);

    CLI11_PARSE(app, argc, argv);

    kamping::Environment env{argc, argv};

    // log level comes from the SPDLOG_LEVEL env var, e.g. SPDLOG_LEVEL=debug
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();

    dss_mehnert::Communicator const comm;
    tlx_die_verbose_unless(
        comm.size() == 1,
        "the generator reproduces a distributed run on a single PE, so it has to be run with a "
        "single MPI rank; use --num-pes to say how many PEs to reproduce"
    );
    tlx_die_verbose_unless(args.num_pes > 0, "--num-pes has to be at least one");

    dss_mehnert::SkewedDNArgs const gen_args{
        .global_strings = args.global_strings(),
        .min_length = args.len_strings_min,
        .max_length = args.len_strings_max,
        .use_uniform_prefix = args.use_uniform_prefix,
        .dn_ratio = args.dn_ratio,
        .skew_fraction = args.skew_fraction,
        .skew_factor = args.skew_factor,
        .placement = clamp_placement(args.id_placement),
        .seed = args.seed,
    };

    auto container = Generator::simulate(gen_args, args.num_pes);
    size_t const num_strings = container.size();

    auto& raw_strings = container.raw_strings();
    size_t const num_lines = to_lines(raw_strings);
    tlx_die_verbose_unless(
        num_lines == num_strings,
        "expected one terminator per string, found " << num_lines << " for " << num_strings
                                                     << " strings"
    );

    std::ofstream out{args.output_path, std::ios::binary};
    tlx_die_verbose_unless(out, "could not open '" << args.output_path << "' for writing");
    out.write(reinterpret_cast<char const*>(raw_strings.data()), std::ssize(raw_strings));
    out.close();
    tlx_die_verbose_unless(out, "could not write to '" << args.output_path << "'");

    SPDLOG_INFO(
        "wrote {} strings ({} bytes) reproducing a run with {} PEs to {}",
        num_strings,
        raw_strings.size(),
        args.num_pes,
        args.output_path
    );

    return 0;
}
