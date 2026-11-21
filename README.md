# dpdetect

## 项目简介

`dpdetect` 是一个基于 OpenCV和ONNX的深度学习推理封装类库，旨在尽可能多对流行的深度学习模型进行[加载模型->输入图片数据->完成推理并解析]这一过程封装。

## 目录结构

- `demo.cpp` - 演示程序入口
- `dp/` - 推理封装类库
- `log/` - 日志模块

## 设计文档

- [`dp` 模块设计说明](dp/DESIGN.md)

## 目前支持

- onnx运行推理
  -  yolov8的检测、实例分割支持，使用"#include <yolov8.h>"
  
### 使用示例

```cpp
#include "yolov8.h"

using namespace dp;

const int classNum = 2;
yolov8OnnxDetector detector(
    classNum,
    std::vector<float>{0.25f, 0.25f});

detector.setDeviceType(OnnxLoader::CPU);
detector.loadModel("model/detect.onnx");

cv::Mat image = cv::imread("test/test.jpg", cv::IMREAD_COLOR);
DetectResArray results = detector.run(std::vector<cv::Mat>{image});

const auto& batch = results.batchAt(0);
const std::vector<DetectRes>& category = batch.classAt(0);
// 或：const auto& category = results.batchClassAt(0, 0);
for (const DetectRes& result : category) {
    // result.boxs / result.scores
}
```

检测和分割结果均按批次、类别、检测项组织。每个 `DetectRes` 包含一个检测框和分数；每个 `SegmentRes` 还包含对应检测框区域的掩膜。访问下标越界时，`batchAt`、`classAt` 和 `batchClassAt` 抛出 `std::out_of_range`。

实例分割使用 `yolov8OnnxSegmenter`，返回 `SegmentResArray`。如模型输入是动态尺寸，应在加载或运行前调用 `setDynamicInputSize` 指定预处理尺寸。完整的设备配置和动态批次示例可参考仓库中的 `demo.cpp`。
