
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

add_library(dp STATIC
    dp.h
    letterbox.h
    dputility.h     dputility.cpp
    yolov8.h
    onnx.h  onnx.cpp
    cvdnn.h
)

target_include_directories(dp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(dp PUBLIC log)
