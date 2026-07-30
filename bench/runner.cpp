// (c) 2019 Matthias Schimek
// (c) 2023 Pascal Mehnert
// This code is licensed under BSD 2-Clause License (see LICENSE for details)

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <kamping/environment.hpp>
#include <kamping/spdlog_adapter/logging.hpp>
#include <spdlog/cfg/env.h>
#include <spdlog/stopwatch.h>
#include <tlx/die/core.hpp>

#include "detail/algorithm_factory.hpp"
#include "detail/cli.hpp"
#include "detail/config_json.hpp"
#include "detail/reporting.hpp"
#include "dss/mpi/communicator.hpp"
#include "dss/util/measuringTool.hpp"

int main(int argc, char* argv[]) {
    SorterArgs args;

    CLI::App app{"a distributed string sorter"};
    app.option_defaults()->always_capture_default();

    std::string timer_json_path;
    std::vector<std::string> levels_param;
    size_t cpus_per_node = 48;
    size_t num_levels = 1;

    add_sorter_args(args, app, timer_json_path, levels_param, cpus_per_node, num_levels);

    CLI11_PARSE(app, argc, argv);

    kamping::Environment env{argc, argv};

    // log level comes from the SPDLOG_LEVEL env var, e.g. SPDLOG_LEVEL=debug
    spdlog::cfg::load_env_levels();
    kamping::logging::setup_logging();

    if (levels_param.size() == 1 && levels_param.front() == "") {
        levels_param.clear();
    }
    parse_level_arg(cpus_per_node, num_levels, levels_param, args.levels);
    set_experiment(args, num_levels);

    dss_mehnert::Communicator comm;

    // the RESULT records go next to the timer JSON, rather than into a redirected stdout
    std::ofstream measurement_file;
    if (!timer_json_path.empty() && comm.is_root()) {
        auto path = std::filesystem::path(timer_json_path);
        if (path.extension() == ".json") {
            path.replace_extension();
        }
        path += "_additional_measurements.txt";

        measurement_file.open(path);
        tlx_die_verbose_unless(measurement_file, "could not open '" << path.string() << "'");
        args.measurement_output = &measurement_file;
    }

    dss_mehnert::Report report;
    auto algo = dss_mehnert::bench::make_algorithm(args, comm);

    auto& measuring_tool = dss_mehnert::measurement::MeasuringTool::measuringTool();
    for (size_t i = 0; i < args.num_iterations; ++i) {
        args.iteration = i;
        measuring_tool.setPrefix(args.get_prefix(comm));
        measuring_tool.setVerbose(args.verbose);

        kamping::measurements::timer().synchronize_and_start("prepare");
        algo->prepare();
        kamping::measurements::timer().stop_and_append();
        kamping::comm_world().barrier();
        spdlog::stopwatch stopwatch;
        kamping::measurements::timer().synchronize_and_start("run");
        algo->run();
        kamping::measurements::timer().stop_and_append();
        SPDLOG_LOGGER_INFO(spdlog::get("root"), "Finished run in {} secs.", stopwatch);
        kamping::measurements::timer().synchronize_and_start("verify");
        algo->verify();
        kamping::measurements::timer().stop_and_append();
        kamping::measurements::timer().synchronize_and_start("report");
        algo->report();
        kamping::measurements::timer().stop_and_append();

        // aggregate this iteration's kamping timer tree and reset it
        report.step_iteration();
    }

    if (!timer_json_path.empty() && comm.is_root()) {
        auto config = dss_mehnert::bench::make_config_json(args, comm, num_levels, cpus_per_node);
        report.push_config(config);

        auto out = dss_mehnert::make_output_stream(timer_json_path);
        report.print(*out);
    }

    return EXIT_SUCCESS;
}
