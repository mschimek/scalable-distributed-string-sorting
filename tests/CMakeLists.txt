FetchContent_Declare(
  KaTestrophe
  GIT_REPOSITORY https://github.com/kamping-site/KaTestrophe.cmake
  GIT_TAG v1.1.1
  SYSTEM EXCLUDE_FROM_ALL)

FetchContent_MakeAvailable(KaTestrophe)

# KaTestrophe::main_thread_single provides the MPI-aware gtest entry point: it initializes MPI,
# makes every rank report its own failures and finalizes MPI once the tests are done.
function(dss_add_mpi_test DSS_TEST_TARGET)
  cmake_parse_arguments(DSS "" "" "FILES;CORES" ${ARGN})
  add_executable(${DSS_TEST_TARGET} ${DSS_FILES})
  target_compile_options(${DSS_TEST_TARGET} PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
  target_include_directories(${DSS_TEST_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${DSS_TEST_TARGET}
    PRIVATE
      dss::dss
      kamping::kamping
      MPI::MPI_CXX
      KaTestrophe::main_thread_single)
  katestrophe_add_mpi_test(${DSS_TEST_TARGET} CORES ${DSS_CORES})
endfunction()

dss_add_mpi_test(test_alltoallv FILES alltoallv_test.cpp CORES 1 2 3 4)
dss_add_mpi_test(test_sampling FILES sampling_test.cpp CORES 1 2 3 4)
dss_add_mpi_test(test_end_to_end FILES end_to_end_test.cpp CORES 1 2 3 4)
dss_add_mpi_test(test_generator FILES generator_test.cpp CORES 1 2 3 4)
dss_add_mpi_test(test_file_generator FILES file_generator_test.cpp CORES 1 2 3 4)
