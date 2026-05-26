#pragma once


#include <onnxruntime_cxx_api.h>


#include "dp.h"

//concurrency并发数，默认0时使用当前可用线程数的一半
inline uint Concurrency(int concurrency) {
	uint threads = std::thread::hardware_concurrency();
	return concurrency == 0 ? std::max(static_cast<uint>(1), threads / 2) : concurrency;
}



//根据中心点坐标和宽高生成矩形框
inline cv::Rect rect(double cx, double cy, double w, double h) {
	return cv::Rect(cv::Point(cx - w / 2, cy - h / 2), cv::Size(w, h));
}

//将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize);

//将矩形框限制在图片范围内
cv::Rect rectValidate(const cv::Rect& box, const cv::Size& size);


/*
* @param oriSize 原图片大小
* @param inputSize 模型输入图片大小
* @param cx，cy，w，h矩形框中心位置x,中心位置y，宽高，对应输入图片中的矩形框
*
* @note 将输入图片中的矩形框转为原始图片中的矩形框
*/
inline cv::Rect oriRect(cv::Size oriSize, cv::Size inputSize, float cx, float cy, float w, float h){
	return scaleRect(rect(cx, cy, w, h), inputSize, oriSize);
}

/*
	@srcMat 输入图像
	@targetSize 目标尺寸
	@autoShape =true将 dw/dh 向下对齐到 stride的整数倍，保证模型下采样时尺寸整除
	@scaleFill =true直接拉伸图像到目标尺寸，不保留宽高比
	@scaleUp =false只缩小不放大
	@stride 对齐值
	@color 填充颜色
*/
class LetterBox {
public:
	LetterBox(cv::Size srcSize, cv::Size targetSize,
		bool autoShape = false, bool scaleFill = false,
		bool scaleUp = false, int stride = 32,
		const cv::Scalar& color = cv::Scalar::all(0));

	void set(cv::Size srcSize, cv::Size targetSize,
		bool autoShape, bool scaleFill,
		bool scaleUp, int stride,
		const cv::Scalar& color);

	cv::Vec4d params() const {
		return _params;
	}

	cv::Mat apply(const cv::Mat& srcMat) const;
	cv::Rect enRect(const cv::Rect& rect) const;

private:
	cv::Size _srcSize;
	cv::Size _targetSize;
	bool _autoShape;
	bool _scaleFill;
	bool _scaleUp;
	int _stride;
	cv::Scalar _color;
	cv::Vec4d _params; // [ratio_x, ratio_y, dw, dh]
};

class SimpleLetterBox : public LetterBox {
public:
	SimpleLetterBox(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color = cv::Scalar::all(0));

	void set(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color);
};

/*
	兼容函数包装器，保留原接口调用方式。
*/
cv::Mat letterBox(const cv::Mat& srcMat, cv::Size targetSize,
	bool autoShape, bool scaleFill, bool scaleUp,
	int stride = 32, const cv::Scalar& color = cv::Scalar::all(0));

inline cv::Mat simpleLetterBox(const cv::Mat& srcMat, cv::Size targetSize,
	const cv::Scalar& color = cv::Scalar::all(0)) {
	SimpleLetterBox box(srcMat.size(), targetSize, color);
	return box.apply(srcMat);
}

cv::Rect enSimpleLetterBoxRect(cv::Size srcSize, cv::Size size, cv::Rect rect);


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
	void load(const char* path, const char* cfg = nullptr);

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
void OnnxLoaderCPU<concurrency>::load(const char* path, const char* cfg)
{
	//Ort环境
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");

	Ort::SessionOptions sessionOptions;
	//设置图形优化级别
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);


	uint count= Concurrency(concurrency);

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


//使用simpleLetterBox调整图片大小
//使用cv::dnn::blobFromImage归一化获取张量
class CVBolbNormalizer {
public:
	CVBolbNormalizer(){}

	cv::Mat operator()(cv::Mat srcMat, cv::Size targetSize, float scalefactor, cv::Scalar mean, bool swapRB);
};


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
	void load(const char* path, const char* cfg = nullptr) override{
		net = cv::dnn::readNetFromONNX(path);

		uint count= Concurrency(concurrency);

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