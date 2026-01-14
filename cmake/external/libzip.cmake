# include(FetchContent)

# set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# FetchContent_Declare(
#     zlib
#     URl https://zlib.net/zlib-1.3.1.tar.gz
# )

# FetchContent_MakeAvailable(zlib)

# add_library(onnxruntime_zlib ALIAS zlib)

# set(ZLIB_FOUND TRUE CACHE INTERNAL "")
# set(ZLIB_LIBRARY zlib CACHE STRING "" FORCE)
# set(ZLIB_INCLUDE_DIR ${zlib_SOURCE_DIR} CACHE INTERNAL "")
# set(ZLIB_LIBRARIES zlib CACHE INTERNAL "")
# set(ZIP_ENABLE_FIND_PACKAGE_ZLIB OFF CACHE BOOL "" FORCE)

# FetchContent_Declare(
#     libzip
#     GIT_REPOSITORY https://github.com/nih-at/libzip.git
#     GIT_TAG v1.10.1
# )

# FetchContent_MakeAvailable(libzip)

# target_link_libraries(zip PRIVATE onnxruntime_zlib)

# set(ZLIB_FOUND FALSE CACHE INTERNAL "")
# set(ZLIB_LIBRARY "" CACHE STRING "" FORCE)
# set(ZLIB_LIBRARIES "" CACHE INTERNAL "")



include(FetchContent)
# ============================================================
# zlib (in-tree, static)
# ============================================================
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(ZLIB_USE_STATIC_LIBS ON CACHE BOOL "" FORCE)
FetchContent_Declare(
 zlib
#  URL https://zlib.net/zlib-1.3.1.tar.gz
 URL file://C:/Users/samrdutt/Downloads/zlib-1.3.1.tar.gz
)
FetchContent_MakeAvailable(zlib)
# ------------------------------------------------------------
# zlib include dirs (source + generated)
# ------------------------------------------------------------
get_target_property(_zlib_src_inc zlib INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _zlib_src_inc)
 set(_zlib_src_inc
   ${zlib_SOURCE_DIR}
   ${zlib_BINARY_DIR}
 )
endif()
# ------------------------------------------------------------
# Provide legacy ZLIB variables (for FindZLIB consumers)
# ------------------------------------------------------------
set(ZLIB_INCLUDE_DIR
 ${zlib_SOURCE_DIR}
 CACHE PATH "" FORCE
)
set(ZLIB_LIBRARY
 zlib
 CACHE STRING "" FORCE
)
set(ZLIB_LIBRARIES
 zlib
 CACHE STRING "" FORCE
)
# ------------------------------------------------------------
# Provide modern imported target EXPECTED by libzip
# ------------------------------------------------------------
if(NOT TARGET ZLIB::ZLIB)
 add_library(ZLIB::ZLIB INTERFACE IMPORTED)
 target_link_libraries(ZLIB::ZLIB INTERFACE zlib)
 target_include_directories(ZLIB::ZLIB INTERFACE
   ${zlib_SOURCE_DIR}
   ${zlib_BINARY_DIR}
 )
endif()
# Private alias for ORT usage
add_library(onnxruntime_zlib ALIAS zlibstatic)
# ============================================================
# libzip
# ============================================================
set(ENABLE_BZIP2 OFF CACHE BOOL "" FORCE)
set(ENABLE_LZMA OFF CACHE BOOL "" FORCE)
set(ENABLE_ZSTD OFF CACHE BOOL "" FORCE)
set(ENABLE_OPENSSL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LIBZIP_DO_INSTALL OFF CACHE BOOL "" FORCE)
set(ENABLE_SHARED OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
 libzip
 GIT_REPOSITORY https://github.com/nih-at/libzip.git
 GIT_TAG v1.10.1
)
FetchContent_MakeAvailable(libzip)
# Ensure libzip links against OUR zlib
target_link_libraries(zip PRIVATE ZLIB::ZLIB)
