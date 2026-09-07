
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_library(log STATIC
    log.cpp
    log.h
)

target_include_directories(log PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(log PUBLIC SPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG)
