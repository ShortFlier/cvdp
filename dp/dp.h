#pragma once

#include <iostream>

#include <opencv2/opencv.hpp>

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
	检测结果结构体
	box：检测框坐标
	score：检测分数
*/
struct DetectRes
{
	cv::Rect box;
	float score;

	DetectRes(){}
	DetectRes(const cv::Rect& rect, float s):box(rect), score(s){}
};
typedef std::vector<std::vector<DetectRes>> DetectResArray;

/*
	分割结果结构体
	box：检测框坐标
	score：检测分数
	mask：分割掩膜，大小为box区域的大小
*/
struct SegmentRes {
	cv::Rect box;
	float score;
	cv::Mat mask;//Mask为box区域的掩膜

	SegmentRes(){}
	SegmentRes(const cv::Rect& box, float score, const cv::Mat& boxMask):box(box), score(score), mask(boxMask){}

};
typedef std::vector<std::vector<SegmentRes>> SegmentResArray;


/*
	深度学习模型加载类
		load，加载模型
		get，返回可用模型
		getSize， 返回模型输入输出尺寸
*/
template<typename _Model>
class ModelLoaderBase{
public:
	virtual void load(const char* path, const char* cfg = nullptr) = 0;
	virtual _Model& get() = 0;
	virtual void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSize) = 0;

	virtual ~ModelLoaderBase(){};
};


/*
	图像归一化器
	  函数符operator()，归一化图像，转为张量
*/
class NormalizerBase{
public:
	virtual cv::Mat operator()(const cv::Mat& src, const cv::Size& targetSize, float scalefactor, const cv::Scalar& mean, bool swapRB) = 0;

	virtual ~NormalizerBase(){};
};


/*
	模型运行器
		函数符operator()，使用模型进行推理，返回结果张量
*/
template<typename _model>
class RunnerBase{
public:
	virtual std::vector<cv::Mat> operator()( _model& model,  cv::Mat& input) = 0;

	virtual ~RunnerBase(){};
};


/*
	结果解析器
		函数符operator()，解析模型输出的结果张量，返回结果数组
*/
template<typename _Result>
class ParserBase{
public:
	virtual _Result operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize,
								const std::vector<std::vector<int>>& outputSizes, int classNum,
								const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs) = 0;

	virtual ~ParserBase(){};
};

/*
	深度学习检测模型功能类

	@_ModelLoader 模型加载器
		参考 ModelLoaderBase 自定义，必须可以调用load，get，getSize函数



	@_Normalizer图像归一化器
		必须是可以调用的函数对象，参考 NormalizerBase


	@_Runner运行器，返回结果张量
		必须是可以调用的函数对象，参考 RunnerBase


	@_Parser解析器，，返回结果数组
		必须是可以调用的函数对象，参考 ParserBase

	@_Result结果数组类型，DetectResArray或SegmentResArray
*/
template<typename _ModelLoader,
	typename _Normalizer,
	typename _Runner,
	typename _Parser,
	typename _Result>
class _DPBase {
	public:
		_ModelLoader _modelLoader;
		_Normalizer _normalizer;
		_Runner _runner;
		_Parser _parser;

		//输入张量大小
		cv::Vec4i _inputSize;
		//输出张量大小
		std::vector<std::vector<int>> _outputSize;

		//类别数
		int _classNum;
		//分数阈值
		std::vector<float> _threshs;
		//NMS阈值
		std::vector<float> _nmsThreshs;

		//归一化参数
		//减去均值
		cv::Scalar _mean = cv::Scalar();
		//缩放因子
		float _scalefactor = 1.0 / 255;
		//交换R、B通道
		bool _swapRB = false;

		


	public:

		//传入的分数阈值或NMS阈值为空，分别设置为默认的0.5、0.4
		//传入的分数阈值或NMS阈值数量不足classNum，使用默认值补全
		_DPBase(int classNum, const std::vector<float>& threshs = std::vector<float>(),
			const std::vector<float>& nmsThreshs = std::vector<float>())
			:_classNum(classNum), _threshs(std::move(threshs)), _nmsThreshs(std::move(nmsThreshs)) {
			for (int i = 0; i < (_classNum - _threshs.size()); ++i){
				_threshs.push_back(0.5);
			}
			for (int i = 0; i < (_classNum - _nmsThreshs.size()); ++i){
				_nmsThreshs.push_back(0.4);
			}

			log_info("DPDetector推理参数classNum: {0}, threshs: {1}, nmsThreshs: {2}",
				 _classNum, fmt::join(_threshs, ", "), fmt::join(_nmsThreshs, ", "));
		}

		//设置归一化参数
		void setNormalizeParam(float scalefactor, const cv::Scalar& mean = cv::Scalar(), bool swapRB = false) {
			_scalefactor = scalefactor;
			_mean = mean;
			_swapRB = swapRB;

			log_info("DPDetector归一化参数scalefactor: {0}, mean: [{1}], swapRB: {2}",
				scalefactor, fmt::join(std::vector<double>{mean[0], mean[1], mean[2]}, ", "), swapRB);
		}

		//加载模型
		void loadModel(const char* path, const char* cfg=nullptr) {
			log_info("DPDetector加载模型: {0}", path);

			_modelLoader.load(path, cfg);
			//获取模型输入大小
			_modelLoader.getSize(_inputSize, _outputSize);

			
			log_info("模型输入大小: [{0},{1},{2},{3}]", _inputSize[0], _inputSize[1], _inputSize[2], _inputSize[3]);
			
			log_info("模型输出{0}: ", _outputSize.size());
			for (size_t i = 0; i < _outputSize.size(); ++i) {
				log_info("======输出{0}: [{1}]", i, fmt::join(_outputSize[i], ","));
			}
		}

		//推理，获取结果
		_Result run(const cv::Mat srcMat) {
			_Result res;

			cv::Size oriSize = srcMat.size();
			log_info("原图大小: [{0}, {1}]", oriSize.width, oriSize.height);

			//归一化
			log_info("执行归一化");
			cv::Size targetSize(_inputSize[3], _inputSize[2]);
			cv::Mat bold = _normalizer(srcMat, targetSize, _scalefactor, _mean, _swapRB);

			//运行
			log_info("执行模型推理");
			std::vector<cv::Mat> resBold;
			ELAPSED("模型推理", resBold = _runner(_modelLoader.get(), bold));

			//解析结果
			log_info("解析结果");
			ELAPSED("结果解析", res = _parser(resBold, oriSize, targetSize, _outputSize, _classNum, _threshs, _nmsThreshs));


			return res;
		}

		_Result run(const char* imgPath) {
			cv::Mat mat = cv::imread(imgPath, cv::IMREAD_UNCHANGED);
			return run(mat);
		}
};

template<typename _ModelLoader,
	typename _Normalizer,
	typename _Runner,
	typename _Parser>
using DPDetector = _DPBase<_ModelLoader, _Normalizer, _Runner, _Parser, DetectResArray>;

template<typename _ModelLoader,
	typename _Normalizer,
	typename _Runner,
	typename _Parser>
using DPSegmentor = _DPBase<_ModelLoader, _Normalizer, _Runner, _Parser, SegmentResArray>;