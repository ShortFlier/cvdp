#pragma once


#include <onnxruntime_cxx_api.h>
#include <windows.h>


#include "dp.h"
#include "letterbox.h"

// 并发数配置。
// concurrency <= 0 时，CPU线程数默认取当前可用线程数的一半；负数仅作为GPU优先标记使用。
inline int Concurrency(int concurrency) {
	int threads = static_cast<int>(std::thread::hardware_concurrency());
	return concurrency <= 0 ? std::max(1, threads / 2) : concurrency;
}


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

namespace detail {
	typedef OrtStatus*(ORT_API_CALL* AppendCudaProviderFn)(OrtSessionOptions*, int);

	inline bool TryAppendCudaExecutionProvider(Ort::SessionOptions& sessionOptions, int deviceId, std::string& errorMessage) {
		HMODULE onnxRuntimeModule = GetModuleHandleW(L"onnxruntime.dll");
		if (onnxRuntimeModule == nullptr) {
			errorMessage = "未找到onnxruntime.dll";
			return false;
		}

		auto appendCuda = reinterpret_cast<AppendCudaProviderFn>(GetProcAddress(onnxRuntimeModule, "OrtSessionOptionsAppendExecutionProvider_CUDA"));
		if (appendCuda == nullptr) {
			errorMessage = "当前onnxruntime.dll未导出CUDA执行提供器接口";
			return false;
		}

		OrtStatus* status = appendCuda(sessionOptions, deviceId);
		if (status != nullptr) {
			auto api = Ort::GetApi();
			errorMessage = api.GetErrorMessage(status);
			api.ReleaseStatus(status);
			return false;
		}

		return true;
	}
}

/*
* onnxruntime的模型加载
* concurrency < 0 时优先尝试GPU推理，失败后自动回退CPU推理
* concurrency = 0 时使用当前可用线程数的一半
* concurrency > 0 时使用指定CPU并发数
*/
template <int concurrency = 0>
class OnnxLoader {

private:
	// Env must outlive Session, so declare Env before Session.
	Ort::Env env;
	Ort::Session session;

public:
	OnnxLoader();

	//加载模型
	void load(const char* path, const char* cfg = nullptr);

	//返回会话
	Ort::Session& get(){
		return session;
	}

	//返回输入、输出张量大小
	void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes);
};
template <int concurrency>
OnnxLoader<concurrency>::OnnxLoader():env(nullptr),session(nullptr)
{
}

template <int concurrency>
void OnnxLoader<concurrency>::load(const char* path, const char* cfg)
{
	(void)cfg;

	//Ort环境
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");

	Ort::SessionOptions sessionOptions;
	//设置图形优化级别
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

	std::string str(path);
	std::wstring wstr(str.begin(), str.end());

	if (concurrency < 0) {
		log_info("Onnx推理模式: GPU优先, 请求参数: {0}", concurrency);

		std::string gpuError;
		if (detail::TryAppendCudaExecutionProvider(sessionOptions, 0, gpuError)) {
			log_info("Onnx推理模式: 已启用CUDA执行提供器, device_id: {0}", 0);
			try {
				session = Ort::Session(env, wstr.c_str(), sessionOptions);
				return;
			}
			catch (const std::exception& e) {
				gpuError = e.what();
				log_warn("Onnx GPU会话创建失败，准备回退CPU推理: {0}", gpuError);
			}
		}
		else {
			log_warn("Onnx GPU执行器初始化失败，准备回退CPU推理: {0}", gpuError);
		}
	}

	int count = Concurrency(concurrency);
	log_info("Onnx推理模式: CPU");
	log_info("Onnx推理并发数: {0}", count);
	sessionOptions.SetIntraOpNumThreads(count);

	session = Ort::Session(env, wstr.c_str(), sessionOptions);
}

template <int concurrency>
void OnnxLoader<concurrency>::getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes)
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