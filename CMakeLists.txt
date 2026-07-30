cmake_minimum_required(VERSION 3.25)
project(distributed_string_sorting)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

include(FetchContent)
FetchContent_Declare(
  kamping
  GIT_REPOSITORY https://github.com/kamping-site/kamping.git
  GIT_TAG v0.2.1
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_Declare(tlx
  GIT_REPOSITORY  https://github.com/tlx/tlx.git
  GIT_TAG         cf83363
  SYSTEM EXCLUDE_FROM_ALL
)

set(CLUSTER_TYPE "SUPERMUCPHASE2")
FetchContent_Declare(rbc
  GIT_REPOSITORY  https://github.com/mschimek/RBC.git
  GIT_TAG         e5556a3
  SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(kamping tlx rbc)

set(IPS4O_DISABLE_PARALLEL ON)
FetchContent_Declare(ips4o
  GIT_REPOSITORY  https://github.com/mschimek/ips4o.git
  GIT_TAG         eb30352
  SYSTEM EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(ips4o)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG 11.0.2
  GIT_SHALLOW ON
  SYSTEM FIND_PACKAGE_ARGS 11.0)

set(SPDLOG_FMT_EXTERNAL_HO ON)
FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog
  GIT_TAG v1.16.0
  GIT_SHALLOW ON
  SYSTEM)

FetchContent_Declare(
  kamping-spdlog-adapter
  GIT_REPOSITORY https://github.com/kamping-site/kamping-spdlog-adapter
  GIT_TAG v1.0.0
  GIT_SHALLOW ON)

FetchContent_Declare(
  kassert
  GIT_REPOSITORY https://github.com/kamping-site/kassert
  GIT_TAG v1.0.0
  SYSTEM EXCLUDE_FROM_ALL)




FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json
  GIT_TAG v3.12.0
  GIT_SHALLOW ON
  SYSTEM)

FetchContent_MakeAvailable(fmt spdlog kamping-spdlog-adapter kassert nlohmann_json)

FetchContent_Declare(
  kaval
  GIT_REPOSITORY https://github.com/niklas-uhl/kaval.git
  GIT_TAG main
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/kaval)

FetchContent_MakeAvailable(kaval)



find_package(TBB REQUIRED)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

option(CLI_ENABLE_PREFIX_DOUBLING "enable prefix doubling merge sort" On)
option(CLI_ENABLE_REDISTRIBUTION "enable support for redistribution strategies" Off)
option(CLI_ENABLE_ALLTOALL "enable support for different AllToAll routines" On)
option(CLI_ENABLE_RQUICK_V1 "enable version of RQuick" Off)
option(CLI_ENABLE_RQUICK_LCP "enable RQuick using LCP values" On)
option(CLI_ENABLE_RQUICK_LONG_FILTER "enable RQuick splitter sort with the long-string filter" On)
option(CLI_ENABLE_ALL "enable all optional command line features" Off)

if(CLI_ENABLE_ALL)
  set(CLI_ENABLE_PREFIX_DOUBLING On)
  set(CLI_ENABLE_REDISTRIBUTION On)
  set(CLI_ENABLE_ALLTOALL On)
  set(CLI_ENABLE_RQUICK_V1 On)
  set(CLI_ENABLE_RQUICK_LCP On)
  set(CLI_ENABLE_RQUICK_LONG_FILTER On)
endif()

message(STATUS "Prefix Doubling Enabled: ${CLI_ENABLE_PREFIX_DOUBLING}")
message(STATUS "Redistribution Strategies Enabled: ${CLI_ENABLE_REDISTRIBUTION}")
message(STATUS "All-To-All Routines Enabled: ${CLI_ENABLE_ALLTOALL}")
message(STATUS "RQuick Version 1 Enabled: ${CLI_ENABLE_RQUICK_V1}")
message(STATUS "RQuick With LCP Enabled: ${CLI_ENABLE_RQUICK_LCP}")
message(STATUS "RQuick Long Filter Enabled: ${CLI_ENABLE_RQUICK_LONG_FILTER}")

option(DSS_BUILD_BENCHMARKS "build the benchmark harness in bench/" ${PROJECT_IS_TOP_LEVEL})
message(STATUS "Benchmarks Enabled: ${DSS_BUILD_BENCHMARKS}")

option(DSS_BUILD_EXAMPLES "build example programs in examples/" Off)
message(STATUS "Examples Enabled: ${DSS_BUILD_EXAMPLES}")

option(DSS_BUILD_TESTS "build the MPI tests in tests/ (run with ctest)" Off)
message(STATUS "Tests Enabled: ${DSS_BUILD_TESTS}")

list(APPEND
  DSS_MEHNERT_WARNING_FLAGS
  "-Wall"
  "-Wextra"
  "-Wundef"
  "-Wunreachable-code"
  "-Wno-unused-parameter"
  "-Wpedantic"
  "-Wnull-dereference"
  "-Wimplicit-fallthrough"
  "-Wno-vla"
  "-Wno-pessimizing-move"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND DSS_MEHNERT_WARNING_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

# kassert #warns and falls back to `normal` unless KASSERT_ASSERTION_LEVEL is defined, so
# it has to be set explicitly per config. 0 = off, 30 = normal; this tracks -DNDEBUG.
# SPDLOG_ACTIVE_LEVEL compiles out SPDLOG_TRACE/SPDLOG_DEBUG below the given level.
set(CMAKE_CXX_FLAGS_DEBUG
    "-ggdb -DKASSERT_ASSERTION_LEVEL=30 -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE")
set(CMAKE_CXX_FLAGS_RELEASE
    "-O3 -DNDEBUG -march=native -DKASSERT_ASSERTION_LEVEL=0 -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "-O2 -DDEBUG -march=native -ggdb -DKASSERT_ASSERTION_LEVEL=30 -DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE")

# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

# Warning flags as a target, so every compiled target picks them up by linking it.
add_library(dss_cxx_flags INTERFACE)
target_compile_options(dss_cxx_flags INTERFACE ${DSS_MEHNERT_WARNING_FLAGS})

add_subdirectory(lib)

if(DSS_BUILD_BENCHMARKS)
  add_subdirectory(bench)
elseif(DSS_BUILD_TESTS)
  # the generator tests exercise the input generators, which live under bench/
  add_subdirectory(bench/input)
endif()

if(DSS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(DSS_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

