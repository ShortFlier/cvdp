#pragma once

#include <onnxruntime_cxx_api.h>

#include "dputility.h"

#include "windows.h"

#include "log.h"

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
