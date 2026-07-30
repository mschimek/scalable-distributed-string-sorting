add_executable(sort_strings_example sort_strings.cpp)

target_link_libraries(sort_strings_example
  PRIVATE
    dss_cxx_flags
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)

add_executable(sort_strings_rquick_splitter_example sort_strings_rquick_splitter.cpp)

target_link_libraries(sort_strings_rquick_splitter_example
  PRIVATE
    dss_cxx_flags
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)

add_executable(sort_strings_rquick_example sort_strings_rquick.cpp)

target_link_libraries(sort_strings_rquick_example
  PRIVATE
    dss_cxx_flags
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)

add_executable(sort_strings_rquick_indexed_example sort_strings_rquick_indexed.cpp)

target_link_libraries(sort_strings_rquick_indexed_example
  PRIVATE
    dss_cxx_flags
    dss::dss
    kamping::kamping
    MPI::MPI_CXX
    kamping::spdlog-adapter
)

add_executable(test_long_filter_equivalence test_long_filter_equivalence.cpp)

target_link_libraries(test_long_filter_equivalence
  PRIVATE
    dss_cxx_flags
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)
