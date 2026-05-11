#pragma once


#include <onnxruntime_cxx_api.h>


#include "dp.h"


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
	Ort::Session& get(){
		return session;
	}

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
void OnnxLoaderCPU<concurrency>::getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes)
{
	//输入大小获取
	auto inputShape=session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	inputSize[0] = inputShape.at(0);
	inputSize[1] = inputShape.at(1);
	inputSize[2] = inputShape.at(2);
	inputSize[3] = inputShape.at(3);

	//输出大小获取
	int outputCount=session.GetOutputCount();
	outputSizes.clear();
	for (int i = 0; i < outputCount; ++i) {
		auto shape=session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		outputSizes.push_back(std::vector<int>(shape.begin(), shape.end()));
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



/*	opencv::dnn模块读取onnx模型的模型读取器
	使用CPU推理
	concurrency并发数，默认0时使用当前可用线程数的一半
*/
typedef cv::dnn::Net DnnNet;

template <uint concurrency= 0>
class CVDnnLoaderCPU: public ModelLoaderBase<DnnNet> {
public:
	CVDnnLoaderCPU(){
		_inputSize=nullptr;
	}
	~CVDnnLoaderCPU(){
		delete _inputSize;
	}

	//加载模型
	void load(const char* path) override{
		net = cv::dnn::readNetFromONNX(path);

		auto threads = std::thread::hardware_concurrency();
		//使用一半的逻辑线程数来运算矩阵
		uint count= concurrency == 0 ? std::max(static_cast<uint>(1), threads / 2) : concurrency;

		// 设置计算后端和线程数
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		cv::setNumThreads(count);

		log_info("opencv::dnn::net推理并发数: {0}", count);
	}

	//返回模型
	DnnNet& get() override{
		return net;
	}

	void setInputSize(int batch, int channel, int height, int width){
		delete _inputSize;
		_inputSize=new cv::Vec4i(batch, channel, height, width);
	}

	//返回输入、输出张量大小
	void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes) override{
		//输入大小获取
		if(_inputSize==nullptr){
			const char* err="opencv::dnn::net无法直接获取输出大小，请先调用setInputSize设置输入大小!";
			log_error(err);
			throw std::runtime_error(err);
		}
		inputSize=*_inputSize;

		//输出大小获取，执行一次前向传播来获取输出大小
		int type=inputSize[1]==3?CV_8UC3:CV_8UC1;
		cv::Mat inputBlob=cv::Mat::zeros(inputSize[2], inputSize[3], type);

		net.setInput(cv::dnn::blobFromImage(inputBlob));
		std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
		std::vector<cv::Mat> outs;
		net.forward(outs, outNames);
		outputSizes.clear();
		for (const auto& out : outs) {
			std::vector<int> shape(out.size.p, out.size.p + out.dims);
			outputSizes.push_back(shape);
		}

	}

private:
	DnnNet net;

	cv::Vec4i* _inputSize;
};



class CVDNNRunner: public RunnerBase<DnnNet> {
public:
	CVDNNRunner() {}

	std::vector<cv::Mat> operator()( DnnNet& net,  cv::Mat& blob) override;
};