# Writes the input of one particular distributed_sorter run to a file, so the very same instance
# can be handed to a shared memory sorter. Its arguments mirror the runner's.
add_executable(string_generator string_generator.cpp)

set_target_properties(string_generator PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}")

target_link_libraries(string_generator
  PRIVATE dss_cxx_flags
          dss_base
          dss_input_generation
          kamping::kamping
          tlx
          CLI11::CLI11
          nlohmann_json::nlohmann_json
          kassert::kassert
          spdlog::spdlog
          kamping::spdlog-adapter)
