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



find_package(TBB REQUIRED)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

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

option(USE_RQUICK_SORT "sort using RQuick alogrithm" Off)
message(STATUS "RQuick Sort Enabled: ${USE_RQUICK_SORT}")

option(DSS_BUILD_EXAMPLES "build example programs in examples/" Off)
message(STATUS "Examples Enabled: ${DSS_BUILD_EXAMPLES}")

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
  "-Wno-vla"
  "-Wno-pessimizing-move"
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  list(APPEND DSS_MEHNERT_WARNING_FLAGS "-Wno-gnu-zero-variadic-macro-arguments")
endif()

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

#set(TLX_USE_LTO ON)
#add_subdirectory(external/tlx)

#add_subdirectory(external/kamping)

# IPS40 doesn't compile wihtout linking to TBB
set(IPS4O_DISABLE_PARALLEL OFF)
#add_subdirectory(external/ips4o)

#set(CLUSTER_TYPE "SUPERMUCPHASE2")
#add_subdirectory(external/RBC)

configure_file("options.hpp.in" "include/options.hpp")

# Create a library containing all implementation, to be linked against by executables
add_library(dss_base)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)
target_compile_options(dss_base PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})

target_include_directories(dss_base PUBLIC
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_include_directories(dss_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dss_base PRIVATE kamping::kamping)
target_link_libraries(dss_base PUBLIC tlx)
target_link_libraries(dss_base PRIVATE MPI::MPI_CXX)
target_link_libraries(dss_base PUBLIC ips4o)
target_link_libraries(dss_base PUBLIC rbc)

add_subdirectory(src)

add_library(dss_lib STATIC src/dss/dss.cpp)
add_library(dss::dss ALIAS dss_lib)

target_include_directories(dss_lib
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include)

target_compile_options(dss_lib PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})

target_link_libraries(dss_lib
  PUBLIC  dss_base
  PRIVATE kamping::kamping MPI::MPI_CXX)

add_executable(distributed_sorter
  src/executables/distributed_sorter.cpp
  src/executables/common_cli.hpp)

target_include_directories(distributed_sorter PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(distributed_sorter PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(distributed_sorter kamping::kamping)
target_link_libraries(distributed_sorter dss_base)
target_link_libraries(distributed_sorter tlx)

add_executable(space_efficient_sorter
  src/executables/space_efficient_sorter.cpp
  src/executables/common_cli.hpp)

target_include_directories(space_efficient_sorter PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/include")

target_compile_options(space_efficient_sorter PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(space_efficient_sorter kamping::kamping)
target_link_libraries(space_efficient_sorter dss_base)
target_link_libraries(space_efficient_sorter tlx)

if(DSS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

