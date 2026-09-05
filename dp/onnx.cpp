#include "onnx.h"

#include "log.h"


namespace detail {

	bool TryAppendCudaExecutionProvider(Ort::SessionOptions& sessionOptions, int deviceId) {
		auto api = Ort::GetApi();
		if (api.SessionOptionsAppendExecutionProvider_CUDA == nullptr) {
			log_error("当前onnxruntime不支持CUDA执行提供器接口");
			return false;
		}

		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

		OrtCUDAProviderOptions cuda_options;
		cuda_options.device_id = deviceId;

		OrtStatus* status = api.SessionOptionsAppendExecutionProvider_CUDA(sessionOptions, &cuda_options);
		if (status != nullptr) {
			auto api = Ort::GetApi();
			log_error("onnx设置CUDA失败：{0}", api.GetErrorMessage(status));
			api.ReleaseStatus(status);
			return false;
		}

		log_info("Onnx推理模式: CUDA，参数: deviceId={0}", deviceId);

		return true;
	}

	bool TryAppendOpenVINOExecutionProvider(Ort::SessionOptions& sessionOptions, int threads, int num_streams) {
		auto api = Ort::GetApi();
		if (api.SessionOptionsAppendExecutionProvider_OpenVINO == nullptr) {
			log_error("当前onnxruntime不支持OpenVINO执行提供器接口");
			return false;
		}

		//关闭onnx高级优化
		sessionOptions.SetGraphOptimizationLevel(ORT_DISABLE_ALL);

		std::unordered_map<std::string, std::string> options;
		options["device_type"] = "CPU";
		options["precision"] = "FP32";
		options["num_of_threads"] = std::to_string(threads);
		options["num_streams"] = std::to_string(num_streams);

		try {
			sessionOptions.AppendExecutionProvider_OpenVINO_V2(options);
		} catch (const std::exception& e) {
			log_error("onnx设置OpenVINO V2失败：{0}", e.what());
			return false;
		}

		log_info("Onnx推理模式: OpenVINO，参数: threads={0}, num_streams={1}", threads, num_streams);

		return true;
	}


	void setCPUSessionOptions(Ort::SessionOptions& sessionOptions, unsigned short intraConcurrency, unsigned short interConcurrency){
		
		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

		//设置矩阵运算内部并发数
		sessionOptions.SetIntraOpNumThreads(intraConcurrency);


		//设置会话并发数
		if (interConcurrency > 0) {
			sessionOptions.SetExecutionMode(ORT_PARALLEL);
			sessionOptions.SetInterOpNumThreads(interConcurrency);
		}

		log_info("Onnx推理模式: CPU, 参数: intraConcurrency={0}, interConcurrency={1}", intraConcurrency, interConcurrency);

	}
}

void OnnxLoader::setSessionOptions(Ort::SessionOptions& sessionOptions){
	bool res=false;

	switch (_deviceType) {
		case NV_CUDA:
			res = detail::TryAppendCudaExecutionProvider(sessionOptions, _cudaDeviceId);
			break;
		case OpenVINO_CPU:
			_openvinoThreads=_openvinoThreads==0?getCPUConcurrency()/2:_openvinoThreads;
			res = detail::TryAppendOpenVINOExecutionProvider(sessionOptions, _openvinoThreads, _openvinoNumStreams);
			break;
	}

	if(!res){
		_intraConcurrency=_intraConcurrency==0?getCPUConcurrency()/2:_intraConcurrency;
		detail::setCPUSessionOptions(sessionOptions, _intraConcurrency, _interConcurrency);
	}
}

void OnnxLoader::loadImpl(const char* path, const char* cfg)
{
	(void)cfg;

	//Ort环境
	env=Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx");

	Ort::SessionOptions sessionOptions;

	std::string str(path);
	std::wstring wstr(str.begin(), str.end());

	setSessionOptions(sessionOptions);

	session = Ort::Session(env, wstr.c_str(), sessionOptions);

}


void OnnxLoader::getSizeImpl(std::vector<std::vector<int>>& inputSize, std::vector<std::vector<int>>& outputSizes)
{
	//输入大小获取
	int inputCount=session.GetInputCount();
	inputSize.clear();
	for (int i = 0; i < inputCount; ++i) {
		auto shape=session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		inputSize.push_back(std::vector<int>(shape.begin(), shape.end()));
	}


	//输出大小获取
	int outputCount=session.GetOutputCount();
	outputSizes.clear();
	for (int i = 0; i < outputCount; ++i) {
		auto shape=session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		outputSizes.push_back(std::vector<int>(shape.begin(), shape.end()));
	}
}






void OnnxRunner::initializeSessionParameters(Ort::Session& session)
{
	const OrtSession* sessionHandle = static_cast<OrtSession*>(session);
	if (_sessionHandle == sessionHandle) {
		return;
	}

	Ort::AllocatorWithDefaultOptions allocator;
	const size_t inputCount = session.GetInputCount();
	const size_t outputCount = session.GetOutputCount();

	_inputNames.resize(inputCount);
	_inputNameArr.resize(inputCount);
	_inputShapes.resize(inputCount);
	for (size_t i = 0; i < inputCount; ++i) {
		_inputNames[i] = session.GetInputNameAllocated(i, allocator).get();
		_inputNameArr[i] = _inputNames[i].c_str();
		_inputShapes[i] = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	}

	_outputNames.resize(outputCount);
	_outputNameArr.resize(outputCount);
	_outputSizes.resize(outputCount);
	for (size_t i = 0; i < outputCount; ++i) {
		_outputNames[i] = session.GetOutputNameAllocated(i, allocator).get();
		_outputNameArr[i] = _outputNames[i].c_str();
		auto shape = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
		_outputSizes[i] = std::vector<int>(shape.begin(), shape.end());
	}

	_sessionHandle = sessionHandle;
}

std::vector<cv::Mat> OnnxRunner::run(Ort::Session& session, std::vector<cv::Mat>& inputDatas)
{
	//获取输入、输出参数
	initializeSessionParameters(session);

	const size_t inputCount = _inputNames.size();
	if (inputDatas.size() != inputCount) {
		std::string errorMsg = "OnnxRunner: 输入张量数量 " + std::to_string(inputDatas.size()) +
			" 与模型输入数量 " + std::to_string(inputCount) + " 不一致";
		log_error(errorMsg);
		throw std::runtime_error(errorMsg);
	}

	const size_t outputCount = _outputNames.size();

	// 按模型输入形状，从对应 cv::Mat 构造 Ort::Value。
	std::vector<Ort::Value> inputValues;
	inputValues.reserve(inputCount);

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	for (size_t i = 0; i < inputCount; ++i) {
		const auto& inputShape = _inputShapes[i];
		// 元素总数
		int64_t inputPixs = 1;
		for (auto d : inputShape) {
			inputPixs *= d;
		}
		inputValues.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, inputDatas[i].ptr<float>(), inputPixs, inputShape.data(), inputShape.size()));
	}

	std::vector<Ort::Value> outputData;
	try{
		outputData = session.Run(Ort::RunOptions(nullptr),
			_inputNameArr.data(), inputValues.data(), inputCount,
			_outputNameArr.data(), outputCount);
	}
	catch (const std::exception& e) {
		std::string errorMsg = "Onnx推理出错: " + std::string(e.what());
		log_error(errorMsg);
		throw std::runtime_error(errorMsg);
	}

	std::vector<cv::Mat> res;
	for(size_t i=0; i<outputCount; ++i) {
		cv::Mat mat(_outputSizes[i].size(), _outputSizes[i].data(), CV_32F, outputData.at(i).GetTensorMutableData<float>());
		res.push_back(mat.clone());
	}

	return res;
}