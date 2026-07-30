// (c) 2023 Pascal Mehnert
// (c) 2026 Matthias Schimek
// This code is licensed under BSD 2-Clause License (see LICENSE for details)
//
// Peels the runtime command line into template arguments. Each algorithm's factory composes
// only the axes it actually uses, and the leaf of the chain returns the constructed algorithm.
// Because lambdas have no name to attach explicit template arguments to, the callbacks are
// invoked as `cb.template operator()<T>()`.

#pragma once

#include <memory>
#include <type_traits>

#include <tlx/die/core.hpp>

#include "bench/algorithm.hpp"
#include "executables/args.hpp"
#include "hash/xxhash.hpp"
#include "mpi/alltoall_strings.hpp"
#include "mpi/communicator.hpp"
#include "options.hpp"
#include "sorter/distributed/bloomfilter.hpp"
#include "sorter/distributed/redistribution.hpp"

namespace dss_mehnert {
namespace bench {

using AlgorithmPtr = std::unique_ptr<AbstractAlgorithm>;

// The alltoallv algorithm is a runtime parameter (see CommonArgs::alltoallv_params), so only
// the two compression flags are peeled into an AlltoallStringsConfig here.
template <typename Callback>
AlgorithmPtr dispatch_alltoall_config(CommonArgs const& args, Callback cb) {
    using dss_mehnert::mpi::AlltoallStringsConfig;

    auto with_flags = [&]<bool compress_lcps, bool compress_prefixes> {
        constexpr AlltoallStringsConfig config{
            .compress_lcps = compress_lcps,
            .compress_prefixes = compress_prefixes,
        };
        using Config = std::integral_constant<AlltoallStringsConfig, config>;
        return cb.template operator()<Config>();
    };

    auto with_lcp_compression = [&]<bool compress_lcps> {
        if (args.prefix_compression) {
            return with_flags.template operator()<compress_lcps, true>();
        } else {
            return with_flags.template operator()<compress_lcps, false>();
        }
    };

    // validates the selected algorithm against the enabled features
    (void)args.alltoallv_params();

    if (args.lcp_compression) {
        return with_lcp_compression.template operator()<true>();
    } else {
        return with_lcp_compression.template operator()<false>();
    }
}

template <typename Callback>
AlgorithmPtr dispatch_bloomfilter(CommonArgs const& args, Callback cb) {
    using namespace dss_mehnert::bloomfilter;

    if (args.grid_bloomfilter) {
        return cb.template operator()<MultiLevel<true, XXHasher>>();
    } else {
        return cb.template operator()<SingleLevel<true, XXHasher>>();
    }
}

// Unlike the other axes this one hands the callback a value, since the redistribution policies
// are constructed here. The row-wise policies are collapsed into one polymorphic type;
// GridwiseRedistribution is passed through directly because it uses a different
// Subcommunicators topology.
template <typename StringSet, typename Callback>
AlgorithmPtr dispatch_redistribution(CommonArgs const& args, Callback cb) {
    using namespace dss_mehnert::redistribution;

    if constexpr (CliOptions::enable_redistribution) {
        using PolymorphicPolicy =
            PolymorphicRedistributionPolicy<StringSet, RowwiseSplit<Communicator>>;

        switch (args.redistribution) {
            case Redistribution::none: {
                // cb(NoRedistribution<Communicator>{});
                tlx_die("disabled for compile-time");
            }
            case Redistribution::naive: {
                return cb(PolymorphicPolicy{NaiveRedistribution<Communicator>{}});
            }
            case Redistribution::simple_strings: {
                return cb(PolymorphicPolicy{SimpleStringRedistribution<Communicator>{}});
            }
            case Redistribution::simple_chars: {
                return cb(PolymorphicPolicy{SimpleCharRedistribution<Communicator>{}});
            }
            case Redistribution::det_strings: {
                return cb(PolymorphicPolicy{DeterministicStringRedistribution<Communicator>{}});
            }
            case Redistribution::det_chars: {
                return cb(PolymorphicPolicy{DeterministicCharRedistribution<Communicator>{}});
            }
            case Redistribution::grid: {
                return cb(GridwiseRedistribution<Communicator>{});
            }
            case Redistribution::sentinel: {
                break;
            }
        }
        tlx_die("unknown redistribution policy");
    } else {
        if (args.redistribution == Redistribution::grid) {
            return cb(GridwiseRedistribution<Communicator>{});
        } else {
            die_with_feature("CLI_ENABLE_REDISTRIBUTION");
        }
    }
}

} // namespace bench
} // namespace dss_mehnert
