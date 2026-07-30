add_subdirectory(alltoallv)

target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         alltoall_strings.hpp
         big_type.hpp
         byte_encoder.hpp
         communicator.hpp
         is_sorted.hpp
         print_strings.hpp
         read_input.hpp)
