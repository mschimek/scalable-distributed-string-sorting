# Benchmark-only dependencies: the command line parser and the JSON report writer. Declared here
# rather than at the top level so that dss_base does not drag them into a consuming project.
FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11
  GIT_TAG v2.5.0
  GIT_SHALLOW ON
  SYSTEM)

FetchContent_Declare(
  kamping-nlohmann-json-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-nlohmann-json-adaptor
  GIT_TAG main
  GIT_SHALLOW ON)

FetchContent_MakeAvailable(cli11 kamping-nlohmann-json-adapter)

add_subdirectory(input)

# One translation unit per algorithm: the AbstractAlgorithm interface keeps each algorithm's
# template tree local to its own TU, so they compile in parallel and an edit to one of them
# does not rebuild the others.
add_executable(distributed_sorter)

target_sources(
  distributed_sorter
  PRIVATE runner.cpp
          detail/algorithm_factory.cpp
          detail/config_json.cpp
          detail/algorithms/merge_sort.cpp
          detail/algorithms/prefix_doubling_simple.cpp
          detail/algorithms/prefix_doubling_multi_level.cpp
          detail/algorithms/rquick.cpp
          detail/algorithms/shared_memory.cpp
  PRIVATE FILE_SET
          HEADERS
          BASE_DIRS
          ${CMAKE_CURRENT_SOURCE_DIR}
          FILES
          detail/algorithm.hpp
          detail/algorithm_factory.hpp
          detail/args.hpp
          detail/cli.hpp
          detail/config_json.hpp
          detail/dispatch.hpp
          detail/enum_names.hpp
          detail/levels.hpp
          detail/stats.hpp
          detail/warmup.hpp
          detail/reporting.hpp
          detail/algorithms/merge_sort.hpp
          detail/algorithms/prefix_doubling.hpp
          detail/algorithms/prefix_doubling_impl.hpp
          detail/algorithms/rquick.hpp
          detail/algorithms/shared_memory.hpp)

# fuzz.py, the kaval command templates and the README all expect the binary at the build root,
# so keep it there rather than in bench/ where the target is defined.
set_target_properties(distributed_sorter PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

target_link_libraries(distributed_sorter
  PRIVATE dss_cxx_flags
          dss_base
          dss_input_generation
          kamping::kamping
          tlx
          CLI11::CLI11
          nlohmann_json::nlohmann_json
          kamping::nlohmann-json-adapter
          kassert::kassert
          spdlog::spdlog
          kamping::spdlog-adapter)

add_subdirectory(tools)
