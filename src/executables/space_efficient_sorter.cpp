// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <map>
#include <numeric>
#include <string>
#include <string_view>
#include <type_traits>

#include <kamping/collectives/barrier.hpp>
#include <kamping/communicator.hpp>
#include <kamping/environment.hpp>
#include <tlx/cmdline_parser.hpp>
#include <tlx/die.hpp>
#include <tlx/die/core.hpp>
#include <tlx/sort/strings/string_ptr.hpp>

#include "executables/common_cli.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "sorter/distributed/space_efficient.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"

enum class CombinedGenerator { none = 0, dn_ratio, sentinel };

enum class CharGenerator { random = 0, file, file_segment, sentinel };

enum class StringGenerator { suffix = 0, window, difference_cover, sentinel };

struct SorterArgs : public CommonArgs {
    bool use_quantile_sampler = false;
    SamplerArgs quantile_sampler;
    size_t combined_gen = static_cast<size_t>(CombinedGenerator::none);
    size_t char_gen = static_cast<size_t>(CharGenerator::random);
    size_t string_gen = static_cast<size_t>(StringGenerator::suffix);
    size_t step = 1;
    size_t num_chars = 100000;
    size_t num_strings = 10000;
    size_t len_strings = 500;
    size_t difference_cover = 3;
    double dn_ratio = 0.5;
    bool shuffle = false;
    std::string path;
    size_t quantile_size = 100 * 1024 * 1024;
    size_t iteration = 0;
    std::vector<size_t> levels;

    // todo print config

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // todo add relevant args
        // clang-format off
        return CommonArgs::get_prefix(comm)
               + " num_chars="        + std::to_string(num_chars)
               + " num_strings="      + std::to_string(num_strings)
               + " len_strings="      + std::to_string(len_strings)
               + " step="             + std::to_string(step)
               + " dn_ratio="         + std::to_string(dn_ratio)
               + " difference_cover=" + std::to_string(difference_cover)
               + " num_levels="       + std::to_string(levels.size())
               + " quantile_size="    + std::to_string(quantile_size)
               + " iteration="        + std::to_string(iteration);
        // clang-format on
    }
};

template <typename StringSet>
auto generate_compressed_strings(SorterArgs const& args, dss_mehnert::Communicator const& comm) {
    using namespace dss_mehnert;

    auto& measuring_tool = measurement::MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_chars = [&]() -> std::vector<typename StringSet::Char> {
        if (args.combined_gen != static_cast<size_t>(CombinedGenerator::none)) {
            return {};
        }

        switch (clamp_enum_value<CharGenerator>(args.char_gen)) {
            case CharGenerator::random: {
                return RandomCharGenerator<StringSet>{args.num_chars};
            }
            case CharGenerator::file: {
                return FileCharGenerator<StringSet>{args.path, comm};
            }
            case CharGenerator::file_segment: {
                return FileSegmentCharGenerator<StringSet>{args.path, args.num_chars, comm};
            }
            case CharGenerator::sentinel: {
                break;
            }
        }
        tlx_die("invalid chararcter generator");
    }();

    auto input_strings = [&]() -> std::vector<typename StringSet::String> {
        if (args.combined_gen != static_cast<size_t>(CombinedGenerator::none)) {
            return {};
        }

        auto const step = args.step, length = args.len_strings;
        auto const dc = args.difference_cover;

        switch (clamp_enum_value<StringGenerator>(args.string_gen)) {
            case StringGenerator::suffix: {
                return CompressedSuffixGenerator<StringSet>{input_chars, step};
            }
            case StringGenerator::window: {
                return CompressedWindowGenerator<StringSet>{input_chars, length, step};
            }
            case StringGenerator::difference_cover: {
                return CompressedDifferenceCoverGenerator<StringSet>{input_chars, dc};
            }
            case StringGenerator::sentinel: {
                break;
            }
        }
        tlx_die("invalid string generator");
    }();

    auto input_container = [&]() mutable -> StringLcpContainer<StringSet> {
        switch (clamp_enum_value<CombinedGenerator>(args.combined_gen)) {
            case CombinedGenerator::none: {
                return {std::move(input_chars), std::move(input_strings)};
            }
            case CombinedGenerator::dn_ratio: {
                return CompressedDNRatioGenerator<StringSet>{
                    args.dn_ratio,
                    args.len_strings,
                    args.num_strings,
                    comm};
            }
            case CombinedGenerator::sentinel: {
                break;
            }
        }
        tlx_die("invalid combined generator");
    }();

    if (args.shuffle) {
        std::random_device rd;
        std::mt19937 gen{rd()};
        auto& strings = input_container.get_strings();
        std::shuffle(strings.begin(), strings.end(), gen);
    }

    measuring_tool.stop("generate_strings");

    comm.barrier();

    auto const num_gen_chars = input_container.char_size();
    auto const num_gen_strs = input_container.size();
    auto const num_uncompressed_chars = input_container.make_string_set().get_sum_length();
    measuring_tool.add(num_gen_chars, "input_chars");
    measuring_tool.add(num_gen_strs, "input_strings");
    measuring_tool.add(num_uncompressed_chars, "uncompressed_input_chars");

    return input_container;
}

template <
    typename CharType,
    typename AlltoallConfig,
    typename RedistributionPolicy,
    typename BloomFilterPolicy>
void run_space_efficient_sort(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;
    namespace sems = dss_mehnert::sorter::space_efficient;

    constexpr auto alltoall_config = AlltoallConfig();

    using StringSet = dss_mehnert::CompressedStringSet<CharType>;
    // todo maybe use separate sample policies
    using PartitionPolicy = dss_mehnert::SpaceEfficientPartitionPolicy<CharType>;
    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using Sorter = sems::SpaceEfficientSort<
        alltoall_config,
        RedistributionPolicy,
        PartitionPolicy,
        PartitionPolicy,
        BloomFilterPolicy>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(args.verbose);

    measuring_tool.disableCommVolume();
    auto input_container = generate_compressed_strings<StringSet>(args, comm);
    measuring_tool.enableCommVolume();

    dss_mehnert::SpaceEfficientChecker<StringSet> checker;
    if (args.check_sorted || args.check_complete) {
        checker.store_container(input_container);
    }

    comm.barrier();

    measuring_tool.start("none", "sorting_overall");
    measuring_tool.start("none", "create_communicators");
    auto const first_level = get_first_level(args.levels, comm);
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    // todo add CLI options for quantile partition policy
    Sorter merge_sort{
        dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
            args.sampler,
            args.get_splitter_sorter()
        ),
        dss_mehnert::init_partition_policy<CharType, PartitionPolicy>(
            args.use_quantile_sampler ? args.quantile_sampler : args.sampler,
            args.get_splitter_sorter()
        ),
        args.quantile_size};
    auto permutation = merge_sort.sort(std::move(input_container), comms);
    measuring_tool.stop("none", "sorting_overall", comm);

    measuring_tool.disable();
    measuring_tool.disableCommVolume();

    if (args.check_sorted) {
        auto const is_sorted = checker.is_sorted(permutation, comm);
        die_verbose_unless(is_sorted, "output permutation is not sorted");
    }
    if (args.check_complete) {
        auto const is_complete = checker.is_complete(permutation, comm);
        die_verbose_unless(is_complete, "output permutation is not complete");
    }

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
}

template <typename... Args>
void run(SorterArgs const& args) {
    dss_mehnert::Communicator comm;

    auto prefix = args.get_prefix(comm);

    // todo print config

    run_space_efficient_sort<Args...>(args, prefix, comm);
}

int main(int argc, char* argv[]) {
    SorterArgs args;

    tlx::CmdlineParser cp;
    cp.set_description("a space efficient distributed string sorter");
    cp.set_author("Pascal Mehnert");

    add_common_args(args, cp);

    cp.add_flag(
        "use-quantile-sampler",
        args.use_quantile_sampler,
        "use separate quantile sampling policy"
    );
    cp.add_flag(
        "quantile-chars",
        args.quantile_sampler.sample_chars,
        "use character based sampling for quantiles"
    );
    cp.add_flag(
        "quantile-indexed",
        args.quantile_sampler.sample_indexed,
        "use indexed sampling for quantiles"
    );
    cp.add_flag(
        "quantile-random",
        args.quantile_sampler.sample_random,
        "use random sampling for quantiles"
    );
    cp.add_size_t(
        "quantile-factor",
        args.quantile_sampler.sampling_factor,
        "use the given oversampling factor for quantiles"
    );

    cp.add_size_t(
        'b',
        "combined-generator",
        args.combined_gen,
        "combined char/string generator to use"
        "([0]=none, 1=dn-ratio)"
    );
    cp.add_size_t(
        'c',
        "char-generator",
        args.char_gen,
        "char generator to use "
        "([0]=random, 1=file, 2=file-segment)"
    );
    cp.add_size_t(
        's',
        "string-generator",
        args.string_gen,
        "string generator to use "
        "([0]=suffix, 1=window, 2=difference_cover)"
    );
    cp.add_size_t('n', "num-strings", args.num_strings, "number of strings per PE");
    cp.add_size_t('m', "len-strings", args.len_strings, "number of characters per string");
    cp.add_size_t('N', "num-chars", args.num_chars, "number of chars per rank");
    cp.add_double('r', "dn-ratio", args.dn_ratio, "D/N ratio of generated strings");
    cp.add_size_t('T', "step", args.step, "characters to skip between strings");
    cp.add_size_t('D', "difference-cover", args.difference_cover, "size of difference cover");
    cp.add_flag("shuffle", args.shuffle, "shuffle the generated strings");
    cp.add_string('y', "path", args.path, "path to input file");
    cp.add_bytes(
        'q',
        "quantile-size",
        args.quantile_size,
        "work on quantiles of the given size [default: 100MiB]"
    );

    std::vector<std::string> levels_param;
    cp.add_opt_param_stringlist(
        "group-size",
        levels_param,
        "size of groups for multi-level merge sort"
    );

    if (!cp.process(argc, argv)) {
        return EXIT_FAILURE;
    }

    parse_level_arg(levels_param, args.levels);

    kamping::Environment env{argc, argv};

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();

    // todo does this make sense? maybe discard first round instead?
    measuring_tool.disableCommVolume();
    mpi_warmup(std::min<size_t>(100000u, args.num_chars), kamping::comm_world());
    measuring_tool.enableCommVolume();

    for (size_t i = 0; i < args.num_iterations; ++i) {
        args.iteration = i;
        dispatch_common_args([&]<typename... T> { run<T...>(args); }, args);
    }

    return EXIT_SUCCESS;
}
