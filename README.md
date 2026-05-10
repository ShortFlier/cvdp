# dpdetect

## 项目简介

`dpdetect` 是一个基于 OpenCV 的深度学习推理封装类库，旨在简化模型加载和推理流程。

该项目支持多种推理方式，包括：

- `opencv::dnn` 模块
- `onnxruntime` C++ 接口

项目设计注重扩展性，方便后续接入更多模型格式或推理框架。

## 主要特性

- 支持 OpenCV DNN 模型加载与推理
- 支持 ONNX Runtime C++ 接口进行 ONNX 模型推理
- 内部日志机制可根据需求自行修改
- 代码结构清晰，便于扩展与定制

## 目录结构

- `demo.cpp` - 演示程序入口
- `dp.h` / `dputility.cpp` / `dputility.h` - 封装的推理功能与工具函数
- `yolov8.cpp` / `yolov8.h` - YOLOv8 相关推理实现
- `log.cpp` / `log.h` - 日志处理模块
- `CMakeLists.txt` - CMake 构建配置

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

项目内部已包含日志模块，方便输出运行信息和调试内容。日志实现可根据项目需求进行替换或扩展。

## 扩展建议

- 添加更多模型格式支持（例如 TensorFlow、TorchScript）
- 集成更多推理框架（例如 TensorRT、OpenVINO）
- 丰富配置与参数管理接口

---

如果你需要进一步定制 `dpdetect`，可以在 `dputility` 和 `yolov8` 模块中拓展推理逻辑、添加更多日志选项。