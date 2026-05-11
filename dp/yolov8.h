#pragma once

#include "dp.h"

#include "dputility.h"





//yolov8检测模型结果解析
class Yolov8DetectResultParser {
public:
	Yolov8DetectResultParser(){}

	DetectResArray operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize, const std::vector<std::vector<int>>& outputSizes,
		int classNum,  const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs);
};

//yolov8分割模型结果解析
class Yolov8SegmentResultParser {
public:
	Yolov8SegmentResultParser(){}

	SegmentResArray operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize, const std::vector<std::vector<int>>& outputSizes,
		int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs);
};

template<uint concurrency= 0>
using yolov8OnnxCPUDetector = DPDetector< OnnxLoaderCPU<concurrency>, CVBolbNormalizer, SingleInputOnnxRunner, Yolov8DetectResultParser>;

template<uint concurrency= 0>
using yolov8OnnxCPUSegmenter = DPSegmentor< OnnxLoaderCPU<concurrency>, CVBolbNormalizer, SingleInputOnnxRunner, Yolov8SegmentResultParser>;

template<uint concurrency= 0>
using yolov8CVDNNCPUDetector= DPDetector< CVDnnLoaderCPU<concurrency>, CVBolbNormalizer, CVDNNRunner, Yolov8DetectResultParser>;

template<uint concurrency= 0>
using yolov8CVDNNCPUSegmenter= DPSegmentor< CVDnnLoaderCPU<concurrency>, CVBolbNormalizer, CVDNNRunner, Yolov8SegmentResultParser>;