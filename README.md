# dpdetect

## 项目简介

`dpdetect` 是一个基于 OpenCV 的深度学习推理封装类库，旨在简化模型加载和推理流程。

该项目支持多种推理方式，包括：

- `opencv::dnn` 模块
- `onnxruntime` C++ 接口

项目设计注重扩展性，方便后续接入更多模型格式或推理框架。

## 新结构说明

本次重构后，项目分为两个子项目：

- `log/` - 日志模块子项目，包含 `log.h` 和 `log.cpp`
- `dp/` - 推理功能子项目，包含 `dp.h`、`dputility.h`、`dputility.cpp`、`yolov8.h`、`yolov8.cpp`

`demo.cpp` 保持在项目根目录，作为程序入口文件。

## 目录结构

- `demo.cpp` - 演示程序入口
- `dp/` - 推理功能子项目
  - `dp.h`
  - `dputility.h`
  - `dputility.cpp`
  - `yolov8.h`
  - `yolov8.cpp`
- `log/` - 日志模块子项目
  - `log.h`
  - `log.cpp`
- `CMakeLists.txt` - 根级别 CMake 配置
- `dp/CMakeLists.txt` - `dp` 子项目配置
- `log/CMakeLists.txt` - `log` 子项目配置

## 快速开始

1. 安装依赖：OpenCV、ONNX Runtime
2. 使用 CMake 生成工程并编译

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

3. 运行演示程序，查看推理效果

## 日志说明

日志模块已拆分为独立子项目 `log/`，方便单独维护和替换。

## 扩展建议

- 添加更多模型格式支持（例如 TensorFlow、TorchScript）
- 集成更多推理框架（例如 TensorRT、OpenVINO）
- 丰富配置与参数管理接口

---

如果你需要进一步定制 `dpdetect`，可以在 `dp/` 子项目中拓展推理逻辑，并在 `log/` 子项目中扩展日志功能。