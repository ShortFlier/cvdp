add_library(dp STATIC
    dp.h
    dputility.h
    dputility.cpp
    yolov8.h
    yolov8.cpp
)

target_include_directories(dp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dp PUBLIC log)
