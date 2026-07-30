# BASE_DIRS of every header file set below, so that the library's headers resolve as
# <dss/...> both in-tree and for a consumer.
set(DSS_LIB_BASE_DIR ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(dss)
