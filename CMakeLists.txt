cmake_minimum_required(VERSION 3.13)

# todo consider moving to C++23
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(distributed_string_sorting)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-Wall -Wextra -Wpedantic)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g")

# Default to Release builds
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPER)
message(STATUS "Build Flags: ${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER}}")

find_package(MPI REQUIRED)
message(STATUS "Run: ${MPIEXEC} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS} ${MPIEXEC_PREFLAGS} EXECUTABLE ${MPIEXEC_POSTFLAGS} ARGS")
# todo target_link_libraries(MyTarget PUBLIC MPI::MPI_CXX)

add_subdirectory(tlx)
add_subdirectory(kamping)

add_library(dss_base)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(dss_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dss_base PUBLIC kamping)
target_link_libraries(dss_base PUBLIC tlx)
target_link_libraries(dss_base PUBLIC MPI::MPI_CXX)

add_subdirectory(src)
