cmake_minimum_required(VERSION 3.12)

project("dpdetect")

enable_language(CXX)

# 如果支持，请为 MSVC 编译器启用热重载。
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()


add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")

#opencv4.8 msvc版本
include_directories("D:\\gw\\opencv\\opencv\\build\\include")
link_directories("D:\\gw\\opencv\\opencv\\build\\x64\\vc16\\lib")
set(OPENCV_WORLD_LIB
    $<$<CONFIG:Debug>:opencv_world480d.lib>
    $<$<NOT:$<CONFIG:Debug>>:opencv_world480.lib>
)

# spdlog 头文件包含形式，可以预编译spdlog库
set(SPDLOG_DIR "D:/gw/clib/spdlog/spdlog-1.15.1/include")
include_directories(${SPDLOG_DIR})

# onnxruntime c++ CPU
# set(ONNX_DLL_DIR "D:/gw/clib/onnxruntime-win-x64-gpu-1.24.1/lib")
# include_directories("D:\\gw\\clib\\onnxruntime-win-x64-gpu-1.24.1\\include")
# link_directories(${ONNX_DLL_DIR})

# # onnxruntime c++ GPU
set(ONNX_DLL_DIR "D:/gw/clib/onnxruntime-win-x64-gpu-1.18.0/lib")
include_directories("D:/gw/clib/onnxruntime-win-x64-gpu-1.18.0/include")
link_directories(${ONNX_DLL_DIR})

project("dpdetect")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(log)
add_subdirectory(dp)

add_executable(dpdetect
    demo.cpp
)

target_include_directories(dpdetect PRIVATE
    ${CMAKE_SOURCE_DIR}/dp
    ${CMAKE_SOURCE_DIR}/log
)

target_link_libraries(dpdetect PRIVATE
    dp
    ${OPENCV_WORLD_LIB}
    onnxruntime.lib
    onnxruntime_providers_shared.lib
)


# 拷贝dll到exe文件目录
add_custom_command(TARGET dpdetect POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_DLL_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:dpdetect>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ONNX_DLL_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:dpdetect>/"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNX_DLL_DIR}/onnxruntime_providers_cuda.dll"
    "$<TARGET_FILE_DIR:dpdetect>/"
    COMMENT "Copying ONNX Runtime DLLs to output directory..."
)
