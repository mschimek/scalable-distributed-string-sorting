target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         lean_non_timer.hpp
         measurements.hpp
         measuringTool.hpp
         non_timer.hpp
         timer.hpp)
