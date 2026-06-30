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
	@autoShape =true将 dw/dh 向下对齐到 stride 的整数倍，保证模型下采样时尺寸整除
	@scaleFill =true直接拉伸图像到目标尺寸，不保留宽高比
	@scaleUp =false只缩小不放大
	@stride 对齐值
	@color 填充颜色

	LetterBox 使用非类型模板参数绑定预处理配置，确保预处理与结果解析使用相同的参数。
*/
template<bool autoShape = false, bool scaleFill = false, bool scaleUp = false, int stride = 32>
class LetterBox {
public:
	LetterBox(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color = cv::Scalar::all(0));

	void set(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color);

	cv::Vec4d params() const {
		return _params;
	}

	cv::Mat apply(const cv::Mat& srcMat) const;
	cv::Rect enRect(const cv::Rect& rect) const;

private:
	cv::Size _srcSize;
	cv::Size _targetSize;
	cv::Scalar _color;
	cv::Vec4d _params; // [ratio_x, ratio_y, dw, dh]
};

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline LetterBox<autoShape, scaleFill, scaleUp, stride>::LetterBox(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
{
	set(srcSize, targetSize, color);
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline void LetterBox<autoShape, scaleFill, scaleUp, stride>::set(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
{
	_srcSize = srcSize;
	_targetSize = targetSize;
	_color = color;

	float r = std::min((float)_targetSize.height / (float)_srcSize.height,
		(float)_targetSize.width / (float)_srcSize.width);
	if (!scaleUp) {
		r = std::min(r, 1.0f);
	}

	float ratio[2] = { r, r };
	int new_un_pad[2] = {
		static_cast<int>(std::round((float)_srcSize.width * r)),
		static_cast<int>(std::round((float)_srcSize.height * r))
	};

	auto dw = static_cast<float>(_targetSize.width - new_un_pad[0]);
	auto dh = static_cast<float>(_targetSize.height - new_un_pad[1]);

	if (autoShape) {
		dw = static_cast<float>(static_cast<int>(dw) % stride);
		dh = static_cast<float>(static_cast<int>(dh) % stride);
	}
	else if (scaleFill) {
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = _targetSize.width;
		new_un_pad[1] = _targetSize.height;
		ratio[0] = static_cast<float>(_targetSize.width) / (float)_srcSize.width;
		ratio[1] = static_cast<float>(_targetSize.height) / (float)_srcSize.height;
	}

	dw /= 2.0f;
	dh /= 2.0f;

	int top = static_cast<int>(std::round(dh - 0.1f));
	int left = static_cast<int>(std::round(dw - 0.1f));

	_params = cv::Vec4d(ratio[0], ratio[1], left, top);
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline cv::Mat LetterBox<autoShape, scaleFill, scaleUp, stride>::apply(const cv::Mat& srcMat) const
{
	if (scaleFill) {
		cv::Mat dst;
		cv::resize(srcMat, dst, _targetSize, 0, 0, cv::INTER_LINEAR);
		return dst;
	}

	float scale = std::min((float)_targetSize.height / (float)_srcSize.height,
		(float)_targetSize.width / (float)_srcSize.width);
	if (!scaleUp) {
		scale = std::min(scale, 1.0f);
	}

	int newW = static_cast<int>(std::round(_srcSize.width * scale));
	int newH = static_cast<int>(std::round(_srcSize.height * scale));

	int padW = _targetSize.width - newW;
	int padH = _targetSize.height - newH;

	if (autoShape) {
		padW = padW / stride * stride;
		padH = padH / stride * stride;
	}

	int padLeft = padW / 2;
	int padTop = padH / 2;

	cv::Mat resized;
	if (_srcSize.width != newW || _srcSize.height != newH) {
		cv::resize(srcMat, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
	}
	else {
		resized = srcMat.clone();
	}

	cv::Mat dst(_targetSize.height, _targetSize.width, srcMat.type(), _color);
	cv::Rect roi(padLeft, padTop, newW, newH);
	resized.copyTo(dst(roi));

	return dst;
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline cv::Rect LetterBox<autoShape, scaleFill, scaleUp, stride>::enRect(const cv::Rect& rect) const
{
	int x = static_cast<int>(std::round((rect.x - _params[2]) / _params[0]));
	int y = static_cast<int>(std::round((rect.y - _params[3]) / _params[1]));
	int width = static_cast<int>(std::round(rect.width / _params[0]));
	int height = static_cast<int>(std::round(rect.height / _params[1]));

	cv::Rect oriRect(x, y, width, height);
	return rectValidate(oriRect, _srcSize);
}

/*
	将预处理器和结果解析器绑定到同一套 LetterBox 参数，避免参数不同步。
*/
template<typename LetterBoxT>
class CVBolbLetterBoxNormalizer {
public:
	CVBolbLetterBoxNormalizer() = default;

	cv::Mat operator()(cv::Mat srcMat, cv::Size targetSize, float scalefactor, cv::Scalar mean, bool swapRB) {
		LetterBoxT box(srcMat.size(), targetSize, cv::Scalar(114, 114, 114));
		cv::Mat mat = box.apply(srcMat);
		cv::Mat blob = cv::dnn::blobFromImage(mat, scalefactor, targetSize, mean, swapRB);
		return blob;
	}
};

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