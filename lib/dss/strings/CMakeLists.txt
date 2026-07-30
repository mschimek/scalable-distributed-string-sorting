target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         bound.hpp
         stringcontainer.hpp
         stringptr.hpp
         stringset.hpp
         stringtools.hpp)
