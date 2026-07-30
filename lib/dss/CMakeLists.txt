# dss_base carries the string sorting implementation. It is header-only, so it has no
# compilable source of its own and needs LINKER_LANGUAGE set explicitly.
add_library(dss_base)
set_target_properties(dss_base PROPERTIES LINKER_LANGUAGE CXX)

# CliOptions: the CLI_ENABLE_* build options as compile-time constants. Generated here rather
# than in bench/ because library headers are gated on it too (see partition_policy_factory.hpp).
configure_file("${PROJECT_SOURCE_DIR}/options.hpp.in"
               "${PROJECT_BINARY_DIR}/include/dss/options.hpp")
target_include_directories(dss_base PUBLIC "${PROJECT_BINARY_DIR}/include")

target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         dss.hpp
         dss_impl.hpp)

add_subdirectory(encoding)
add_subdirectory(hash)
add_subdirectory(merge)
add_subdirectory(mpi)
add_subdirectory(sorter)
add_subdirectory(strings)
add_subdirectory(util)

target_link_libraries(dss_base PRIVATE kamping::kamping)
target_link_libraries(dss_base PRIVATE MPI::MPI_CXX)
target_link_libraries(dss_base PUBLIC tlx)
target_link_libraries(dss_base PUBLIC ips4o)
target_link_libraries(dss_base PUBLIC rbc)
# PUBLIC: headers under lib/dss/ use KASSERT and SPDLOG_*, so every consumer needs them
target_link_libraries(dss_base PUBLIC kassert::kassert)
target_link_libraries(dss_base PUBLIC spdlog::spdlog kamping::spdlog-adapter)
target_compile_features(dss_base PUBLIC cxx_std_20)

# dss::dss is the facade: dss.cpp holds the explicit instantiations of the run_* entry points
# declared in dss.hpp, so a consumer does not pay for the template tree.
add_library(dss_lib STATIC dss.cpp)
add_library(dss::dss ALIAS dss_lib)

target_link_libraries(dss_lib
  PUBLIC  dss_base
  PRIVATE dss_cxx_flags kamping::kamping MPI::MPI_CXX)
