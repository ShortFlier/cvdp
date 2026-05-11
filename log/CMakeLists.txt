add_library(log STATIC
    log.cpp
    log.h
)

target_include_directories(log PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
