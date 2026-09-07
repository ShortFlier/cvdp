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


void OnnxLoader::getSizeImpl(std::vector<TensorInfo>& inputSize, std::vector<TensorInfo>& outputSizes)
{
	Ort::AllocatorWithDefaultOptions allocator;

	//输入大小获取
	int inputCount=session.GetInputCount();
	inputSize.resize(inputCount);
	for (int i = 0; i < inputCount; ++i) {
		inputSize[i].name=session.GetInputNameAllocated(i, allocator).get();
		inputSize[i].shape=session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	}


	//输出大小获取
	int outputCount=session.GetOutputCount();
	outputSizes.resize(outputCount);
	for (int i = 0; i < outputCount; ++i) {
		outputSizes[i].name=session.GetOutputNameAllocated(i, allocator).get();
		outputSizes[i].shape=session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
	}
}






std::vector<Tensor> OnnxRunner::run(Ort::Session& session, const std::vector<Tensor>& inputDatas)
{
	//默认内存分配器
	Ort::AllocatorWithDefaultOptions allocator;

	auto inputCount = inputDatas.size();
	if ( session.GetInputCount() != inputCount) {
		std::string errorMsg = "OnnxRunner: 输入张量数量 " + std::to_string(inputDatas.size()) +
			" 与模型输入数量 " + std::to_string(inputCount) + " 不一致";
		log_error(errorMsg);
		return std::vector<Tensor>();
	}


	// 输入信息：按模型声明的每个输入，从对应 cv::Mat 构造 Ort::Value
	std::vector<Ort::Value> inputValues;
	inputValues.reserve(inputCount);
	std::vector<const char*> inputNameArr(inputCount);

	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

	for (int i = 0; i < inputCount; ++i) {
		inputNameArr[i] = inputDatas[i].info.name.c_str();

		auto& inputShape = inputDatas[i].info.shape;
		// 元素总数
		int64_t inputPixs = 1;
		for (auto d : inputShape) {
			inputPixs *= d;
		}
		//转输入张量
		inputValues.push_back(Ort::Value::CreateTensor<float>(
			memoryInfo, const_cast<float*>(inputDatas[i].tensor.ptr<float>()), inputPixs, inputShape.data(), inputShape.size()));
	}

	// 输出信息
	auto outputCount = session.GetOutputCount();

	std::vector<Tensor> outputTensors(outputCount);
	std::vector<const char*> outputNameArr(outputCount);
	for(int i=0; i<outputCount; ++i) {
		outputTensors[i].info.name = session.GetOutputNameAllocated(i, allocator).get();

		outputNameArr[i] = outputTensors[i].info.name.c_str();
	}

	std::vector<Ort::Value> outputData;
	try{
		outputData = session.Run(Ort::RunOptions(nullptr),
			inputNameArr.data(), inputValues.data(), inputCount,
			outputNameArr.data(), outputCount);
	}
	catch (const std::exception& e) {
		std::string errorMsg = "Onnx推理出错: " + std::string(e.what());
		log_error(errorMsg);
		return std::vector<Tensor>();
	}

	for(int i=0; i<outputCount; ++i) {
		//获取输出张量的形状
		outputTensors[i].info.shape = outputData[i].GetTensorTypeAndShapeInfo().GetShape();

		// 将输出张量转换为 cv::Mat
		std::vector<int> outputShape(outputTensors[i].info.shape.begin(), outputTensors[i].info.shape.end());
		cv::Mat mat(outputShape, CV_32F, outputData[i].GetTensorMutableData<float>());
		
		outputTensors[i].tensor = mat.clone();
	}

	return outputTensors;
}