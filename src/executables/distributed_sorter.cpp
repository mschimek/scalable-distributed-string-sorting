// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
#include "mpi/alltoall.hpp"
#include "mpi/communicator.hpp"
#include "mpi/is_sorted.hpp"
#include "mpi/warmup.hpp"
#include "options.hpp"
#include "sorter/distributed/merge_sort.hpp"
#include "sorter/distributed/prefix_doubling.hpp"
#include "strings/stringset.hpp"
#include "util/measuringTool.hpp"
#include "util/string_generator.hpp"

enum class StringGenerator {
    skewed_random = 0,
    dn_ratio = 1,
    file = 2,
    skewed_dn_ratio = 3,
    suffix = 4,
    sentinel = 5
};

struct SorterArgs : public CommonArgs {
    size_t string_generator = static_cast<size_t>(StringGenerator::dn_ratio);
    size_t num_strings = 100000;
    size_t len_strings = 100;
    size_t len_strings_min = len_strings;
    size_t len_strings_max = len_strings + 10;
    std::string path;
    double DN_ratio = 0.5;
    size_t iteration = 0;
    bool strong_scaling = false;
    std::vector<size_t> levels;

    std::string get_prefix(dss_mehnert::Communicator const& comm) const {
        // clang-format off
        return CommonArgs::get_prefix(comm) 
               + " num_strings="    + std::to_string(num_strings)
               + " len_strings="    + std::to_string(len_strings)
               + " num_levels="     + std::to_string(levels.size())
               + " iteration="      + std::to_string(iteration)
               + " strong_scaling=" + std::to_string(strong_scaling)
               + " dn_ratio="       + std::to_string(DN_ratio);
        // clang-format on
    }
};

template <typename StringSet>
auto generate_strings(SorterArgs const& args, dss_mehnert::Communicator const& comm) {
    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();

    comm.barrier();
    measuring_tool.start("generate_strings");

    auto input_container = [=]() -> dss_mehnert::StringLcpContainer<StringSet> {
        using namespace dss_schimek;

        auto const num_strings = (args.strong_scaling ? 1 : comm.size()) * args.num_strings;
        auto const len_strings = args.len_strings;
        auto const DN_ratio = args.DN_ratio;
        auto const& path = args.path;

        switch (clamp_enum_value<StringGenerator>(args.string_generator)) {
            case StringGenerator::skewed_random: {
                tlx_die("not implemented");
            }
            case StringGenerator::dn_ratio: {
                return DNRatioGenerator<StringSet>{num_strings, len_strings, DN_ratio};
            }
            case StringGenerator::file: {
                check_path_exists(path);
                return FileDistributer<StringSet>{path};
            }
            case StringGenerator::skewed_dn_ratio: {
                return SkewedDNRatioGenerator<StringSet>{num_strings, len_strings, DN_ratio};
            }
            case StringGenerator::suffix: {
                check_path_exists(path);
                return SuffixGenerator<StringSet>{path};
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

    return input_container;
}

template <
    typename CharType,
    typename PartitionPolicy,
    typename MPIAllToAllRoutine,
    typename RedistributionPolicy,
    typename LcpCompression,
    typename PrefixCompression,
    typename BloomFilterPolicy>
void run_merge_sort(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;

    constexpr bool lcp_compression = LcpCompression();
    constexpr bool prefix_compression = PrefixCompression();

    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
    using AllToAllPolicy = dss_mehnert::mpi::
        AllToAllStringImpl<lcp_compression, prefix_compression, MPIAllToAllRoutine>;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using MergeSort = dss_mehnert::sorter::
        DistributedMergeSort<RedistributionPolicy, AllToAllPolicy, PartitionPolicy>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    measuring_tool.disableCommVolume();
    auto input_container = generate_strings<StringSet>(args, comm);
    auto num_input_chars = input_container.char_size();
    auto num_input_strs = input_container.size();

    CheckerWithCompleteExchange<StringLcpPtr> checker;
    if (args.check || args.check_exhaustive) {
        checker.storeLocalInput(input_container.raw_strings());
    }
    measuring_tool.enableCommVolume();

    comm.barrier();

    measuring_tool.start("none", "sorting_overall");
    measuring_tool.start("none", "create_communicators");
    auto const first_level = get_first_level(args.levels, comm);
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    MergeSort merge_sort{PartitionPolicy{args.sampling_factor}};
    merge_sort.sort(input_container, comms);
    measuring_tool.stop("none", "sorting_overall", comm);

    if (args.check || args.check_exhaustive) {
        auto sorted_str_ptr = input_container.make_string_lcp_ptr();
        auto num_sorted_chars = input_container.char_size();
        auto num_sorted_strs = input_container.size();

        bool is_sorted = is_complete_and_sorted(
            sorted_str_ptr,
            num_input_chars,
            num_sorted_chars,
            num_input_strs,
            num_sorted_strs,
            comm
        );
        die_verbose_unless(is_sorted, "output is not sorted or missing characters");

        if (args.check_exhaustive) {
            bool const is_complete = checker.check(sorted_str_ptr, true, comm);
            die_verbose_unless(is_complete, "output is not a permutation of the input");
        }
    }

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
}

template <
    typename CharType,
    typename PartitionPolicy,
    typename MPIAllToAllRoutine,
    typename RedistributionPolicy,
    typename LcpCompression,
    typename PrefixCompression,
    typename BloomFilterPolicy>
void run_prefix_doubling(
    SorterArgs const& args, std::string prefix, dss_mehnert::Communicator const& comm
) {
    using namespace dss_schimek;

    constexpr bool lcp_compression = LcpCompression();
    constexpr bool prefix_compression = PrefixCompression();

    using StringSet = dss_mehnert::StringSet<CharType, dss_mehnert::Length>;
    using StringLcpPtr = tlx::sort_strings_detail::StringLcpPtr<StringSet, size_t>;
    using AllToAllPolicy = dss_mehnert::mpi::
        AllToAllStringImpl<lcp_compression, prefix_compression, MPIAllToAllRoutine>;

    using Subcommunicators = RedistributionPolicy::Subcommunicators;
    using MergeSort = dss_mehnert::sorter::PrefixDoublingMergeSort<
        RedistributionPolicy,
        AllToAllPolicy,
        PartitionPolicy,
        BloomFilterPolicy>;

    using dss_mehnert::measurement::MeasuringTool;
    auto& measuring_tool = MeasuringTool::measuringTool();
    measuring_tool.setPrefix(prefix);
    measuring_tool.setVerbose(false);

    measuring_tool.disableCommVolume();
    auto input_container = generate_strings<StringSet>(args, comm);
    auto num_input_chars = input_container.char_size();
    auto num_input_strs = input_container.size();

    CheckerWithCompleteExchange<StringLcpPtr> checker;
    if (args.check || args.check_exhaustive) {
        checker.storeLocalInput(input_container.raw_strings());
    }
    measuring_tool.enableCommVolume();

    comm.barrier();
    measuring_tool.start("none", "sorting_overall");
    measuring_tool.start("none", "create_communicators");
    auto const first_level = get_first_level(args.levels, comm);
    Subcommunicators comms{first_level, args.levels.end(), comm};
    measuring_tool.stop("none", "create_communicators", comm);

    MergeSort merge_sort{PartitionPolicy{args.sampling_factor}};
    auto permutation = merge_sort.sort(std::move(input_container), comms);
    measuring_tool.stop("none", "sorting_overall", comm);

    measuring_tool.disable();
    measuring_tool.disableCommVolume();

    if ((args.check || args.check_exhaustive) && comm.size() > 1) {
        StringLcpContainer<StringSet> local_input{checker.local_input()};
        auto complete_strings_cont = dss_mehnert::sorter::apply_permutation(
            local_input.make_string_set(),
            permutation,
            comm
        );

        auto sorted_str_ptr = complete_strings_cont.make_string_lcp_ptr();
        auto num_sorted_chars = complete_strings_cont.char_size();
        auto num_sorted_strs = complete_strings_cont.size();

        bool is_complete_and_sorted = dss_schimek::is_complete_and_sorted(
            sorted_str_ptr,
            num_input_chars,
            num_sorted_chars,
            num_input_strs,
            num_sorted_strs,
            comm
        );

        die_verbose_unless(is_complete_and_sorted, "output is not sorted or missing characters");

        if (args.check_exhaustive) {
            die_verbose_unless(
                checker.check(sorted_str_ptr, false, comm),
                "output is not a permutation of the input"
            );
        }
    }

    measuring_tool.write_on_root(std::cout, comm);
    measuring_tool.reset();
}

template <typename... Args>
void run(SorterArgs const& args) {
    dss_mehnert::Communicator comm;

    auto prefix = args.get_prefix(comm);

    // todo print config

    if (args.prefix_doubling) {
        if constexpr (CliOptions::enable_prefix_doubling) {
            run_prefix_doubling<Args...>(args, prefix, comm);
        } else {
            die_with_feature("CLI_ENABLE_PREFIX_DOUBLING");
        }
    } else {
        run_merge_sort<Args...>(args, prefix, comm);
    }
}


int main(int argc, char* argv[]) {
    SorterArgs args;

    tlx::CmdlineParser cp;
    cp.set_description("a distributed string sorter");
    cp.set_author("Matthias Schimek, Pascal Mehnert");

    add_common_args(args, cp);

    cp.add_size_t(
        'k',
        "generator",
        args.string_generator,
        "type of string generation to use "
        "(0=skewed, [1]=DNGen, 2=file, 3=skewedDNGen, 4=suffixGen)"
    );
    cp.add_string('y', "path", args.path, "path to input file");
    cp.add_double('r', "DN-ratio", args.DN_ratio, "D/N ratio of generated strings");
    cp.add_size_t('n', "num-strings", args.num_strings, "number of strings to be generated");
    cp.add_size_t('m', "len-strings", args.len_strings, "length of generated strings");
    cp.add_size_t(
        'b',
        "min-len-strings",
        args.len_strings_min,
        "minimum length of generated strings"
    );
    cp.add_size_t(
        'B',
        "max-len-strings",
        args.len_strings_max,
        "maximum length of generated strings"
    );
    cp.add_flag('x', "strong-scaling", args.strong_scaling, "perform a strong scaling experiment");

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
    auto num_bytes = std::min<size_t>(args.num_strings * 5, 100000u);
    dss_schimek::mpi::randomDataAllToAllExchange(num_bytes);
    measuring_tool.enableCommVolume();


    for (size_t i = 0; i < args.num_iterations; ++i) {
        args.iteration = i;
        dispatch_common_args([&]<typename... T> { run<T...>(args); }, args);
    }

    return EXIT_SUCCESS;
}
