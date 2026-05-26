#include "dputility.h"



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

cv::Rect rectValidate(const cv::Rect& box, const cv::Size& size){
	cv::Rect validBox = box;
	//宽高至少为1
	validBox.width = std::max(validBox.width, 1);
	validBox.height = std::max(validBox.height, 1);

	cv::Rect imgRect(0, 0, size.width, size.height);
	return validBox & imgRect; // 取交集，确保在图片范围内
}

LetterBox::LetterBox(cv::Size srcSize, cv::Size targetSize,
	bool autoShape, bool scaleFill,
	bool scaleUp, int stride,
	const cv::Scalar& color)
{
	set(srcSize, targetSize, autoShape, scaleFill, scaleUp, stride, color);
}

void LetterBox::set(cv::Size srcSize, cv::Size targetSize,
	bool autoShape, bool scaleFill,
	bool scaleUp, int stride,
	const cv::Scalar& color)
{
	_srcSize = srcSize;
	_targetSize = targetSize;
	_autoShape = autoShape;
	_scaleFill = scaleFill;
	_scaleUp = scaleUp;
	_stride = stride;
	_color = color;

	float r = std::min((float)_targetSize.height / (float)_srcSize.height,
		(float)_targetSize.width / (float)_srcSize.width);
	if (!_scaleUp) {
		r = std::min(r, 1.0f);
	}

	float ratio[2] = { r, r };
	int new_un_pad[2] = {
		static_cast<int>(std::round((float)_srcSize.width * r)),
		static_cast<int>(std::round((float)_srcSize.height * r))
	};

	auto dw = static_cast<float>(_targetSize.width - new_un_pad[0]);
	auto dh = static_cast<float>(_targetSize.height - new_un_pad[1]);

	if (_autoShape) {
		dw = static_cast<float>(static_cast<int>(dw) % _stride);
		dh = static_cast<float>(static_cast<int>(dh) % _stride);
	}
	else if (_scaleFill) {
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

cv::Mat LetterBox::apply(const cv::Mat& srcMat) const
{
	if (_scaleFill) {
		cv::Mat dst;
		cv::resize(srcMat, dst, _targetSize, 0, 0, cv::INTER_LINEAR);
		return dst;
	}

	float scale = std::min((float)_targetSize.height / (float)_srcSize.height,
		(float)_targetSize.width / (float)_srcSize.width);
	if (!_scaleUp) {
		scale = std::min(scale, 1.0f);
	}

	int newW = static_cast<int>(std::round(_srcSize.width * scale));
	int newH = static_cast<int>(std::round(_srcSize.height * scale));

	int padW = _targetSize.width - newW;
	int padH = _targetSize.height - newH;

	if (_autoShape) {
		padW = padW / _stride * _stride;
		padH = padH / _stride * _stride;
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

cv::Rect LetterBox::enRect(const cv::Rect& rect) const
{
	int x = static_cast<int>(std::round((rect.x - _params[2]) / _params[0]));
	int y = static_cast<int>(std::round((rect.y - _params[3]) / _params[1]));
	int width = static_cast<int>(std::round(rect.width / _params[0]));
	int height = static_cast<int>(std::round(rect.height / _params[1]));

	cv::Rect oriRect(x, y, width, height);
	return rectValidate(oriRect, _srcSize);
}

SimpleLetterBox::SimpleLetterBox(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
	: LetterBox(srcSize, targetSize, false, false, false, 32, color)
{
}

void SimpleLetterBox::set(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
{
	LetterBox::set(srcSize, targetSize, false, false, false, 32, color);
}

cv::Mat letterBox(const cv::Mat& srcMat, cv::Size targetSize, bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar& color){
	LetterBox box(srcMat.size(), targetSize, autoShape, scaleFill, scaleUp, stride, color);
	return box.apply(srcMat);
}

cv::Rect enSimpleLetterBoxRect(cv::Size srcSize, cv::Size size, cv::Rect rect){
	LetterBox box(srcSize, size, false, false, false, 32, cv::Scalar::all(0));
	return box.enRect(rect);
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
		res.push_back(mat);
	}

	return res;
}

cv::Mat CVBolbNormalizer::operator()(cv::Mat srcMat, cv::Size targetSize, float scalefactor, cv::Scalar mean, bool swapRB)
{
	cv::Mat mat=simpleLetterBox(srcMat, targetSize, cv::Scalar(114, 114, 114));
	cv::Mat blob=cv::dnn::blobFromImage(mat, scalefactor, targetSize, mean, swapRB);
	return blob;
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