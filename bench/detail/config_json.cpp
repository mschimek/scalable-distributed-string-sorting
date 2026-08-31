// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include "detail/config_json.hpp"

#include <cstdlib>
#include <string>

namespace dss_mehnert {
namespace bench {

namespace {

// The MPI collective-tuning knobs are set in the environment by the job script, not on our command
// line, so record what the run actually saw. Missing means the launcher left the default in place.
std::string env_or_empty(char const* name) {
    char const* val = std::getenv(name);
    return val == nullptr ? std::string{} : std::string{val};
}

} // namespace

nlohmann::ordered_json make_config_json(
    SorterArgs const& args,
    Communicator const& comm,
    std::size_t num_levels,
    std::size_t cpus_per_node
) {
    nlohmann::ordered_json config;
    config["p"] = comm.size();
    config["experiment"] = args.experiment;
    config["i_mpi_adjust_alltoallv"] = env_or_empty("I_MPI_ADJUST_ALLTOALLV");
    config["i_mpi_adjust_allgatherv"] = env_or_empty("I_MPI_ADJUST_ALLGATHERV");
    config["ompi_mca_coll_tuned_use_dynamic_rules"] =
        env_or_empty("OMPI_MCA_coll_tuned_use_dynamic_rules");
    config["ompi_mca_coll_tuned_alltoallv_algorithm"] =
        env_or_empty("OMPI_MCA_coll_tuned_alltoallv_algorithm");

    config["input"]["string-generator"] = args.string_generator;
    config["input"]["path"] = args.path;
    config["input"]["max-num-bytes"] = args.max_num_bytes;
    config["input"]["num-strings"] = args.num_strings;
    config["input"]["length-strings"] = args.len_strings;
    config["input"]["min-len-strings"] = args.len_strings_min;
    config["input"]["max-len-strings"] = args.len_strings_max;
    config["input"]["DN-ratio"] = args.dn_ratio;
    config["input"]["dn-encode-padding"] = args.dn_encode_padding;
    config["input"]["use-uniform-prefix"] = args.use_uniform_prefix;
    config["input"]["skew-fraction"] = args.skew_fraction;
    config["input"]["skew-factor"] = args.skew_factor;
    config["input"]["placement"] = args.id_placement;
    // the run being reproduced; `p` above is 1 for a simulated run, so record it separately
    config["input"]["simulate-num-pes"] = args.simulate_num_pes;

    config["num-iterations"] = args.num_iterations;
    config["mpi-warmup-rounds"] = args.mpi_warmup_rounds;
    config["algorithm"] = args.algorithm;
    config["permutation"] = args.permutation;
    config["num-levels"] = num_levels;
    config["cpus-per-node"] = cpus_per_node;
    config["group-size"] = args.levels;

    config["sample-chars"] = args.sampler.sample_chars;
    config["sample-indexed"] = args.sampler.sample_indexed;
    config["sample-random"] = args.sampler.sample_random;
    config["sampling-factor"] = args.sampler.sampling_factor;
    config["splitter-length-factor"] = args.sampler.splitter_length_factor;
    config["redistribute-sample"] = args.sampler.redistribute_sample;
    config["level-adjusted-scaling"] = args.sampler.level_adjusted_scaling;
    config["local-sorter"] = args.local_sorter;
    config["splitter-sequential"] = args.splitter_sequential;

    config["rquick-v1"] = args.rquick_v1;
    config["rquick-lcp"] = args.rquick_lcp;
    config["long-filter"] = args.long_filter;
    // kept for backwards compatibility with existing result-processing scripts; derived from
    // `algorithm` now that prefix doubling is one of several `--algorithm` choices
    config["prefix-doubling"] = args.algorithm == Algorithm::prefix_doubling;
    config["grid-bloomfilter"] = args.grid_bloomfilter;
    config["bloomfilter-base-case"] = args.bloomfilter_base_case;
    config["bloomfilter-level-dedup"] = args.bloomfilter_level_dedup;
    config["lcp-compression"] = args.lcp_compression;
    config["prefix-compression"] = args.prefix_compression;
    config["alltoall"] = args.alltoall_algorithm;
    config["alltoall_large_counts"] = args.alltoall_large_counts;
    config["alltoall_onefactor_num_slots"] = args.onefactor_num_slots;
    config["alltoall_onefactor_synchronized"] = args.onefactor_synchronized;
    config["alltoall_onefactor_use_issend"] = args.onefactor_use_issend;
    config["redistribution"] = enum_name(redistribution_names, args.redistribution);
    config["strong-scaling"] = args.strong_scaling;

    config["check-sorted"] = args.check_sorted;
    config["check-complete"] = args.check_complete;
    config["count-prefixes"] = args.count_prefixes;
    config["print-sorted"] = args.print_sorted;
    config["gather-counters"] = args.gather_counters;
    config["verbose"] = args.verbose;

    return config;
}

} // namespace bench
} // namespace dss_mehnert
