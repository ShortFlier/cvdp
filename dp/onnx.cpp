#include "onnx.h"

#include "log.h"


namespace detail {

	bool TryAppendCudaExecutionProvider(Ort::SessionOptions& sessionOptions, int deviceId) {
		auto api = Ort::GetApi();
		if (api.SessionOptionsAppendExecutionProvider_CUDA == nullptr) {
			log_error("当前onnxruntime不支持CUDA执行提供器接口");
			return false;
		}

		sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);

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

void OnnxLoader::load(const char* path, const char* cfg)
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