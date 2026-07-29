add_subdirectory(alltoallv)

target_sources(dss_base
    PUBLIC
        alltoall_strings.hpp
        big_type.hpp
        byte_encoder.hpp
        communicator.hpp
        is_sorted.hpp
        read_input.hpp
)
