#include "dputility.h"

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
		res.push_back(mat);
	}

	return res;
}


cv::Mat CVBolbNormalizer::operator()(cv::Mat srcMat, cv::Size targetSize, float scalefactor, cv::Scalar mean, bool swapRB)
{
	return cv::dnn::blobFromImage(srcMat, scalefactor, targetSize, mean, swapRB);
}

cv::Rect oriRect(cv::Size oriSize, cv::Size inputSize, float cx, float cy, float w, float h)
{
	return scaleRect(rect(cx, cy, w, h), inputSize, oriSize);
}

//将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize) {
	double x_scale = static_cast<double>(toSize.width) / fromSize.width;
	double y_scale = static_cast<double>(toSize.height) / fromSize.height;

	int x = static_cast<int>(box.x * x_scale);
	int y = static_cast<int>(box.y * y_scale);
	int width = static_cast<int>(box.width * x_scale);
	int height = static_cast<int>(box.height * y_scale);

	return cv::Rect(x, y, width, height);
}


std::vector<cv::Mat> CVDNNRunner::operator()( DnnNet& net,  cv::Mat& blob) {
	// 设置输入数据
	net.setInput(blob);

	//获取输出层
	std::vector<std::string> outputLayerNames = net.getUnconnectedOutLayersNames();

	// 执行推理
	std::vector<cv::Mat> outputs;
	net.forward(outputs, outputLayerNames);

	return outputs;
}