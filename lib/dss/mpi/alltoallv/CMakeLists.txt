target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         direct.hpp
         dispatch.hpp
         log.hpp
         native.hpp
         onefactor.hpp
         pairwise.hpp
         params.hpp)
