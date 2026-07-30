# Input handling is a benchmark concern, not a sorting one: the generators and the file reader
# live here so that dss_base carries no notion of where its strings come from.
# Header-only, hence INTERFACE.
add_library(dss_input_generation INTERFACE)

target_sources(
  dss_input_generation
  INTERFACE FILE_SET
            HEADERS
            BASE_DIRS
            ${CMAKE_CURRENT_LIST_DIR}/..
            FILES
            input.hpp
            string_generator.hpp)

# nlohmann_json is pulled in transitively: input.hpp takes the parsed SorterArgs, which carries
# the JSON name tables from detail/enum_names.hpp.
target_link_libraries(dss_input_generation
  INTERFACE dss_base
            kamping::kamping
            tlx
            kassert::kassert
            nlohmann_json::nlohmann_json
            spdlog::spdlog
            kamping::spdlog-adapter)
