cmake_minimum_required(VERSION 3.13)

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)

project(distributed_string_sorting)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

option(CLI_ENABLE_PREFIX_DOUBLING "enable prefix doubling merge sort" On)
option(CLI_ENABLE_REDISTRIBUTION "enable support for redistribution strategies" Off)
option(CLI_ENABLE_ALLTOALL "enable support for different AllToAll routines" Off)
option(CLI_ENABLE_RQUICK_V1 "enable version of RQuick" Off)
option(CLI_ENABLE_RQUICK_LCP "enable RQuick using LCP values" On)
option(CLI_ENABLE_ALL "enable all optional command line features" Off)

if(CLI_ENABLE_ALL)
  set(CLI_ENABLE_PREFIX_DOUBLING On)
  set(CLI_ENABLE_REDISTRIBUTION On)
  set(CLI_ENABLE_ALLTOALL On)
  set(CLI_ENABLE_RQUICK_V1 On)
  set(CLI_ENABLE_RQUICK_LCP On)
endif()

message(STATUS "Prefix Doubling Enabled: ${CLI_ENABLE_PREFIX_DOUBLING}")
message(STATUS "Redistribution Strategies Enabled: ${CLI_ENABLE_REDISTRIBUTION}")
message(STATUS "All-To-All Routines Enabled: ${CLI_ENABLE_ALLTOALL}")
message(STATUS "RQuick Version 1 Enabled: ${CLI_ENABLE_RQUICK_V1}")
message(STATUS "RQuick With LCP Enabled: ${CLI_ENABLE_RQUICK_LCP}")

option(USE_SHARED_MEMORY_SORT "sort using a shared memory string sorting algorithm" Off)
message(STATUS "Shared Memory Enabled: ${USE_SHARED_MEMORY_SORT}")

list(APPEND
  DSS_MEHNERT_WARNING_FLAGS
  "-Werror"
  "-Wall"
  "-Wextra"
  "-Wundef"
  "-Wunreachable-code"
  "-Wno-unused-parameter"
  "-Wpedantic"
  "-Wnull-dereference"
  "-Wimplicit-fallthrough"

  # todo remove vla usages
  "-Wno-vla"

  # todo remove if possible
  "-Wno-pessimizing-move"

  # todo produce too much noise with existing code
  # "-Wshadow"
  # "-Wconversion"
  # "-Wsign-conversion"
  # "-Wunused"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND DSS_MEHNERT_WARNING_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

## todo maybe use -flto, but consider binary size (maybe strip?)
set(CMAKE_CXX_FLAGS_DEBUG "-ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -DDEBUG -march=native -ggdb")

# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")

set(TLX_USE_LTO ON)
add_subdirectory(external/tlx)

add_subdirectory(external/kamping)

# IPS40 doesn't compile wihtout linking to TBB
set(IPS4O_DISABLE_PARALLEL OFF)
add_subdirectory(external/ips4o)

# todo propbably don't want to set this globally
set(CLUSTER_TYPE "SUPERMUCPHASE2")
add_subdirectory(external/RBC)

configure_file("options.hpp.in" "include/options.hpp")

# Create a library containing all implementation, to be linked against by executables
add_library(dss_base)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(dss_base PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})

target_include_directories(dss_base PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_include_directories(dss_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dss_base PRIVATE kamping)
target_link_libraries(dss_base PUBLIC tlx)
target_link_libraries(dss_base PRIVATE MPI::MPI_CXX)
target_link_libraries(dss_base PUBLIC ips4o)
target_link_libraries(dss_base PUBLIC rbc)

add_subdirectory(src)

add_executable(distributed_sorter
  src/executables/distributed_sorter.cpp
  src/executables/common_cli.hpp)

target_include_directories(distributed_sorter PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(distributed_sorter PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(distributed_sorter kamping)
target_link_libraries(distributed_sorter dss_base)
target_link_libraries(distributed_sorter tlx)

add_executable(space_efficient_sorter
  src/executables/space_efficient_sorter.cpp
  src/executables/common_cli.hpp)

target_include_directories(space_efficient_sorter PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(space_efficient_sorter PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(space_efficient_sorter kamping)
target_link_libraries(space_efficient_sorter dss_base)
target_link_libraries(space_efficient_sorter tlx)

