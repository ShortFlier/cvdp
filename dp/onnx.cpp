#include "onnx.h"

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


void OnnxLoader::load(const char* path, const char* cfg)
{
	(void)cfg;

	//Ort环境
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "yolo");

	Ort::SessionOptions sessionOptions;
	//设置图形优化级别
	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

	std::string str(path);
	std::wstring wstr(str.begin(), str.end());

	if (_usingGPU) {
		log_info("Onnx推理模式: GPU");

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

	
	log_info("Onnx推理模式: CPU");

	//设置矩阵运算内部并发数
	if (_intraConcurrency <= 0) {
		_intraConcurrency = getCPUConcurrency();
	}
	log_info("Onnx推理内部并发数: {0}", _intraConcurrency);
	sessionOptions.SetIntraOpNumThreads(_intraConcurrency);


	//设置会话并发数
	if (_interConcurrency > 0) {
		log_info("Onnx推理会话并发数: {0}", _interConcurrency);
		sessionOptions.SetExecutionMode(ORT_PARALLEL);
		sessionOptions.SetInterOpNumThreads(_interConcurrency);
	}

	session = Ort::Session(env, wstr.c_str(), sessionOptions);

}


void OnnxLoader::getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes)
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






std::vector<cv::Mat> SingleInputOnnxRunner::operator()(Ort::Session& session, cv::Mat blob)
{
	//默认内存分配器
	Ort::AllocatorWithDefaultOptions allocator;

	int inputCount = session.GetInputCount();
	if (inputCount != 1) {
		std::string errorMsg = "SingleInputOnnxRunner只接受单输入模型，但当前模型输入数量为: " + std::to_string(inputCount);
		log_error(errorMsg);
		throw std::runtime_error(errorMsg);
	}

	int outputCount = session.GetOutputCount();

	// 输入信息
	std::string inputName = session.GetInputNameAllocated(0, allocator).get();
	auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
	int64 inputSize[4] = { inputShape.at(0), inputShape.at(1), inputShape.at(2), inputShape.at(3) };
	int64 inputPixs = inputSize[1] * inputSize[2] * inputSize[3];

	// 输出信息
	std::string outputName = session.GetOutputNameAllocated(0, allocator).get();

	std::vector<std::string> outputNames(outputCount);
	std::vector<std::vector<int>> outputSizes(outputCount);
	for(int i=0; i<outputCount; ++i) {
		outputNames[i] = session.GetOutputNameAllocated(i, allocator).get();
		auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		outputSizes[i] = std::vector<int>(shape.begin(), shape.end());
	}

	// 输入数据
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	auto inputData = Ort::Value::CreateTensor<float>(memoryInfo, blob.ptr<float>(), inputPixs, inputSize, 4);

	std::array<const char*, 1> inputNameArr = { inputName.c_str() };
	std::vector<const char*> outputNameArr(outputCount);
	for(int i=0; i<outputCount; ++i) {
		outputNameArr[i] = outputNames[i].c_str();
	}

	std::vector<Ort::Value> outputData;
	try{
		outputData = session.Run(Ort::RunOptions(nullptr), inputNameArr.data(), &inputData, 1, outputNameArr.data(), outputCount);
	}
	catch (const std::exception& e) {
		std::string errorMsg = "Onnx推理出错: " + std::string(e.what());
		log_error(errorMsg);
		throw std::runtime_error(errorMsg);
	}

	std::vector<cv::Mat> res;
	for(int i=0; i<outputCount; ++i) {
		cv::Mat mat(outputSizes[i].size(), outputSizes[i].data(), CV_32F, outputData.at(i).GetTensorMutableData<float>());
		// Copy out tensor data, because outputData will be released when this function returns.
		res.push_back(mat.clone());
	}

	return res;
}