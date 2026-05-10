#pragma once

#include <opencv2/opencv.hpp>


#include <onnxruntime_cxx_api.h>

#include "log.h"


#include <chrono>


#define ELAPSED(str, exe)                                                     \
    do {                                                                      \
        auto __start = std::chrono::high_resolution_clock::now();             \
        exe;                                                                   \
        auto __end = std::chrono::high_resolution_clock::now();               \
        auto __duration = std::chrono::duration_cast<std::chrono::microseconds>(__end - __start).count(); \
        log_info("{0}耗时：{1}ms", str, __duration / 1000.0); \
    } while(0)

/*
*onnxruntime的模型加载
*使用CPU推理
*concurrency并发数，默认0时使用当前可用线程数的一半
*/
template <uint concurrency= 0>
class OnnxLoaderCPU {

private:
	Ort::Session session;
	Ort::Env env;

public:
	OnnxLoaderCPU();

	//加载模型
	void load(const char* path);

	//返回会话
	Ort::Session& get();

	//返回输入、输出张量大小
	void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes);
};
template <uint concurrency>
OnnxLoaderCPU<concurrency>::OnnxLoaderCPU():session(nullptr),env(nullptr)
{
}

template <uint concurrency>
void OnnxLoaderCPU<concurrency>::load(const char* path)
{
	//Ort环境
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");

	Ort::SessionOptions sessionOptions;
	//设置图形优化级别
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);


	//设置线程数
	auto threads = std::thread::hardware_concurrency();
	//使用一半的逻辑线程数来运算矩阵
	uint count= concurrency == 0 ? std::max(static_cast<uint>(1), threads / 2) : concurrency;

	log_info("Onnx推理并发数: {0}", count);

	sessionOptions.SetIntraOpNumThreads(count);

	std::string str(path);
	std::wstring wstr(str.begin(), str.end());
	session = Ort::Session(env, wstr.c_str(), sessionOptions);
}

template <uint concurrency>
Ort::Session& OnnxLoaderCPU<concurrency>::get()
{
	return session;
}

template <uint concurrency>
void OnnxLoaderCPU<concurrency>::getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes)
{
	//输入大小获取
	auto inputShape=session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	inputSize[0] = inputShape.at(0);
	inputSize[1] = inputShape.at(1);
	inputSize[2] = inputShape.at(2);
	inputSize[3] = inputShape.at(3);

	log_info("模型输入大小: [{0},{1},{2},{3}]", inputSize[0], inputSize[1], inputSize[2], inputSize[3]);

	//输出大小获取
	int outputCount=session.GetOutputCount();
	outputSizes.clear();
	for (int i = 0; i < outputCount; ++i) {
		auto shape=session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		outputSizes.push_back(std::vector<int>(shape.begin(), shape.end()));
	}

	log_info("模型输出大小: ");
	for (size_t i = 0; i < outputSizes.size(); ++i) {
		log_info("输出{0}: [{1}]", i, fmt::join(outputSizes[i], ","));
	}
}



//onnxruntime的单输入运行推理
class SingleInputOnnxRunner {
public:
	SingleInputOnnxRunner() {}

	std::vector<cv::Mat> operator()(Ort::Session& session, cv::Mat blob);
};


//使用cv::dnn::blobFromImage归一化获取张量
class CVBolbNormalizer {
public:
	CVBolbNormalizer(){}

	cv::Mat operator()(cv::Mat srcMat, cv::Size targetSize, float scalefactor, cv::Scalar mean, bool swapRB);
};

/*
* @param oriSize 原图片大小
* @param inputSize 模型输入图片大小
* @param cx，cy，w，h矩形框中心位置x,中心位置y，宽高，对应输入图片中的矩形框
*
* @note 将输入图片中的矩形框转为原始图片中的矩形框
*/
cv::Rect oriRect(cv::Size oriSize, cv::Size inputSize, float cx, float cy, float w, float h);


//根据中心点坐标和宽高生成矩形框
inline cv::Rect rect(double cx, double cy, double w, double h) {
	return cv::Rect(cv::Point(cx - w / 2, cy - h / 2), cv::Size(w, h));
}

//将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize);