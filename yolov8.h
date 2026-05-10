#pragma once

#include "dp.h"

#include "dputility.h"





//yolov8检测模型结果解析
class Yolov8DetectResultParser {
public:
	Yolov8DetectResultParser(){}

	DetectResArray operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize, std::vector<std::vector<int>> outputSizes,
		int classNum, std::vector<float>& socreThreshs, std::vector<float>& nmsThreshs);
};

//yolov8分割模型结果解析
class Yolov8SegmentResultParser {
public:
	Yolov8SegmentResultParser(){}

	SegmentResArray operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize, std::vector<std::vector<int>> outputSizes,
		int classNum, std::vector<float>& socreThreshs, std::vector<float>& nmsThreshs);
};

template<uint concurrency= 0>
using yolov8OnnxCPUDetector = DPDetector< OnnxLoaderCPU<concurrency>, CVBolbNormalizer, SingleInputOnnxRunner, Yolov8DetectResultParser>;

template<uint concurrency= 0>
using yolov8OnnxCPUSegmenter = DPSegmentor< OnnxLoaderCPU<concurrency>, CVBolbNormalizer, SingleInputOnnxRunner, Yolov8SegmentResultParser>;