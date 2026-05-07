add_executable(sort_strings_example sort_strings.cpp)

target_compile_options(sort_strings_example PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(sort_strings_example
  PRIVATE
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)

add_executable(sort_strings_rquick_example sort_strings_rquick.cpp)

target_compile_options(sort_strings_rquick_example PRIVATE ${DSS_MEHNERT_WARNING_FLAGS})
target_link_libraries(sort_strings_rquick_example
  PRIVATE
    dss::dss
    kamping::kamping
    MPI::MPI_CXX)
