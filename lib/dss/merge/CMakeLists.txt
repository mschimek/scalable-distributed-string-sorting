target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         bingmann-lcp_losertree.hpp
         lcp_merge.hpp
         stringptr.hpp
         stringtools.hpp)
