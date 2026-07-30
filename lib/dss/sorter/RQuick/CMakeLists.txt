target_sources(
  dss_base
  PUBLIC FILE_SET
         HEADERS
         BASE_DIRS
         ${DSS_LIB_BASE_DIR}
         FILES
         BinTreeMedianSelection.hpp
         RandomBitStore.hpp
         RQuick.hpp)
