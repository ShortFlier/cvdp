#pragma once

#include "dp.h"

#include "letterbox.h"

#include "onnx.h"

#include "cvdnn.h"



//yolov8检测模型结果解析
template<typename LetterBoxT, int index>
class Yolov8DetectLetterBoxResultParser:public ParserBase<Yolov8DetectLetterBoxResultParser<LetterBoxT, index>, DetectResArray>{
public:
	Yolov8DetectLetterBoxResultParser(){}

	DetectResArray parse(std::vector<cv::Mat>& outputs, cv::Size oriSize, const std::vector<std::vector<int>>& inputSize, const std::vector<std::vector<int>>& outputSizes,
		int classNum,  const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs);

private:
	LetterBoxT _letterBox;
};


template<typename LetterBoxT, int index>
DetectResArray Yolov8DetectLetterBoxResultParser<LetterBoxT, index>::parse(std::vector<cv::Mat>& outputs, cv::Size oriSize, const std::vector<std::vector<int>>& _inputSize,
	const std::vector<std::vector<int>>& outputSizes, int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs)
{
	DetectResArray resArr(classNum);

	//模型图片输入大小
	cv::Size inputSize=getImageInputSize(_inputSize, index);

	//只有一个输出
	cv::Mat& outputMat = outputs.at(0);
	const std::vector<int>& outputSize = outputSizes.at(0);

	//yolov8输出格式[1，类别属性，预测数]
	outputMat=outputMat.reshape(1, outputSize.at(0) * outputSize.at(1));

	std::vector<std::vector<float>> scores(classNum);
	std::vector<std::vector<cv::Rect>> boxs(classNum);
	//矩形框格式为cx，cy，w，h
	for (int c = 0; c < outputMat.cols; ++c) {
		float cx = outputMat.at<float>(0, c);
		float cy = outputMat.at<float>(1, c);
		float w = outputMat.at<float>(2, c);
		float h = outputMat.at<float>(3, c);

		cv::Rect box = rect(cx, cy, w, h);

		for (int i = 0; i < classNum; ++i) {
			boxs[i].push_back(box);
			scores[i].push_back(outputMat.at<float>(4 + i, c));
		}
	}

	for (int i = 0; i < classNum; ++i) {
		std::vector<int> indexs;
		cv::dnn::NMSBoxes(boxs[i], scores[i], socreThreshs.at(i), nmsThreshs.at(i), indexs);

		for (int j = 0; j < indexs.size(); ++j) {
			int index = indexs[j];

			//对应原图矩形框大小
			_letterBox.set(oriSize, inputSize, cv::Scalar(114, 114, 114));
			cv::Rect box = _letterBox.enRect(boxs[i][index]);

			resArr[i].push_back(DetectRes(box, scores[i][index]));
		}
	}

	return resArr;
}






//yolov8分割模型结果解析
template<typename LetterBoxT, int index>
class Yolov8SegmentLetterBoxResultParser:public ParserBase<Yolov8SegmentLetterBoxResultParser<LetterBoxT, index>, SegmentResArray>{
public:
	Yolov8SegmentLetterBoxResultParser(){}

	SegmentResArray parse(std::vector<cv::Mat>& outputs, cv::Size oriSize, const std::vector<std::vector<int>>& _inputSize, const std::vector<std::vector<int>>& outputSizes,
		int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs);

private:
	LetterBoxT _letterBox;
};


template<typename LetterBoxT, int index>
SegmentResArray Yolov8SegmentLetterBoxResultParser<LetterBoxT, index>::parse(std::vector<cv::Mat>& outputs, cv::Size oriSize, const std::vector<std::vector<int>>& _inputSize,
	 const std::vector<std::vector<int>>& outputSizes,	int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs)
{
	SegmentResArray resArr(classNum);

	//模型图片输入大小
	cv::Size inputSize=getImageInputSize(_inputSize, index);

	/*
	outputs应该有两个输出张量
	一个是特征输出[1, 预测信息，预测数]，预测信息格式[cx, cy, w, h, class1_score, ..., ceof]
	另一个是原型掩膜特征图[1,ceof, h, w]
	*/
	if(outputs.size() != 2) {
		std::string errMsg = "Yolov8SegmentLetterBoxResultParser::operator() error: outputs size should be 2, but get " + std::to_string(outputs.size());
		log_error(errMsg);
		throw std::runtime_error(errMsg);
	}

	/*
		获取每个类别分数，进行NMS操作，初步筛选
	*/	
	std::vector<cv::Rect> outputBoxs;
	std::vector<std::vector<float>> scores(classNum);

	cv::Mat predMat=outputs[0].reshape(1, outputSizes[0][0] * outputSizes[0][1]);
	//每一列向量是一个预测,[cx, cy, w, h, class1_score, ..., ceof]
	for (int c = 0; c < predMat.cols; ++c) {
		float cx = predMat.at<float>(0, c);
		float cy = predMat.at<float>(1, c);
		float w = predMat.at<float>(2, c);
		float h = predMat.at<float>(3, c);

		cv::Rect box = rect(cx, cy, w, h);
		box = rectValidate(box, inputSize);
		outputBoxs.push_back(box);

		for (int i = 0; i < classNum; ++i) {
			scores[i].push_back(predMat.at<float>(4 + i, c));
		}

	}

	//NMS操作
	std::vector<std::vector<int>> classIndexs(classNum);
	for (int i = 0; i < classNum; ++i) {
		cv::dnn::NMSBoxes(outputBoxs, scores[i], socreThreshs.at(i), nmsThreshs.at(i), classIndexs[i]);
	}

	/*
	根据NMS结果，获取对应的掩膜特征图，生成分割掩膜
	*/
	cv::Mat mask_protos=outputs[1];
	//mask_protos信息
	int seg_c=outputSizes[1][1];
	int seg_h=outputSizes[1][2];
	int seg_w=outputSizes[1][3];

	_letterBox.set(oriSize, inputSize, cv::Scalar(114, 114, 114));
	cv::Vec4d params = _letterBox.params();

	for(int i=0; i<classNum; ++i) {
		for(int j=0; j<classIndexs[i].size(); ++j) {
			int index=classIndexs[i][j];

			/*
				提取掩膜特征图对应区域，计算分割掩膜，减少计算量
			*/
			cv::Rect oriRect = _letterBox.enRect(outputBoxs[index]);
			int net_width = inputSize.width;
			int net_height = inputSize.height;


			//计算mask_protos对应的区域
			int rang_x = static_cast<int>(std::floor((oriRect.x * params[0] + params[2]) / net_width * seg_w));
			int rang_y = static_cast<int>(std::floor((oriRect.y * params[1] + params[3]) / net_height * seg_h));
			int rang_w = static_cast<int>(std::ceil(((oriRect.x + oriRect.width) * params[0] + params[2]) / net_width * seg_w)) - rang_x;
			int rang_h = static_cast<int>(std::ceil(((oriRect.y + oriRect.height) * params[1] + params[3]) / net_height * seg_h)) - rang_y;

			rang_w = std::max(rang_w, 1);
			rang_h = std::max(rang_h, 1);
			if (rang_x + rang_w > seg_w) {
				if (seg_w - rang_x > 0)
					rang_w = seg_w - rang_x;
				else
					rang_x -= 1;
			}
			if (rang_y + rang_h > seg_h) {
				if (seg_h - rang_y > 0)
					rang_h = seg_h - rang_y;
				else
					rang_y -= 1;
			}

			std::vector<cv::Range> ranges;
			ranges.push_back(cv::Range(0, 1));
			ranges.push_back(cv::Range::all());
			ranges.push_back(cv::Range(rang_y, rang_y + rang_h));
			ranges.push_back(cv::Range(rang_x, rang_x + rang_w));


			//提取对应区域的mask_protos，运算掩膜
			cv::Mat temp_mask_protos = mask_protos(ranges).clone();
			temp_mask_protos = temp_mask_protos.reshape(0, {seg_c, rang_w * rang_h});

			cv::Mat ceof=predMat.col(index).rowRange(4 + classNum, predMat.rows).t();

			cv::Mat mask_feature = ceof * temp_mask_protos;
			mask_feature = mask_feature.reshape(0, rang_h);

			cv::Mat dest;
			cv::exp(-mask_feature, dest);
			dest = 1.0 / (1.0 + dest);


			//mask_protos区域映射到原图区域
			int left = static_cast<int>(std::floor((net_width / static_cast<double>(seg_w) * rang_x - params[2]) / params[0]));
			int top = static_cast<int>(std::floor((net_height / static_cast<double>(seg_h) * rang_y - params[3]) / params[1]));
			int width = static_cast<int>(std::ceil(net_width / static_cast<double>(seg_w) * rang_w / params[0]));
			int height = static_cast<int>(std::ceil(net_height / static_cast<double>(seg_h) * rang_h / params[1]));

			cv::Mat maskPatch;
			cv::resize(dest, maskPatch, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
			cv::Mat finalMask = maskPatch(oriRect - cv::Point(left, top));

			cv::Mat oriMask;
			finalMask.convertTo(oriMask, CV_8UC1, 255);
			resArr[i].push_back(SegmentRes(oriRect,  scores[i][index], oriMask));

		}
	}

	return resArr;
}



template<bool autoShape = false, bool scaleFill = false, bool scaleUp = false, int stride = 32, int index=0>
class LetterBoxConfig {
public:
	using Box = LetterBox<autoShape, scaleFill, scaleUp, stride>;
	using Normalizer = LetterBoxNormalizer<Box, index>;
	using DetectParser = Yolov8DetectLetterBoxResultParser<Box, index>;
	using SegmentParser = Yolov8SegmentLetterBoxResultParser<Box, index>;
};

using SimpleLetterBoxConfig = LetterBoxConfig<false, false, false, 32, 0>;

template<typename Yolov8OnnxDP>
class Yolov8OnnxDPImpl:public Yolov8OnnxDP{
public:
	//使用父类构造函数	
	using Yolov8OnnxDP::Yolov8OnnxDP;

	/*
	//扁平化接口
	*/

	//设置使用推理设备
	void setDeviceType(OnnxLoader::DeviceType deviceType){
		this->_modelLoader.setDeviceType(deviceType);
	}
	
	/*
	* 设置使用cpu推理时的参数，设置intra并发数优于设置inter并发数
	*@intraConcurrency 设置CPU推理时，intra并发数，设置0时，设置为当前CPU线程数一半
	*@interConcurrency 设置CPU推理时，inter并发数，0时不设置inter并发
	*/
	void setCPUParams(unsigned short intraConcurrency, unsigned short interConcurrency=0) {
		this->_modelLoader.setCPUParams(intraConcurrency, interConcurrency);
	}

	/*
	* 设置OpenVINO推理时的参数
	*@threads，设置OpenVINO推理时的线程数，设置为0时，使用当前CPU线程数一半
	*@num_streams，设置OpenVINO推理时的流数，设置为0时，使用默认值
	*/
	void setOpenVINOCPUParams(unsigned short threads, unsigned short num_streams=0) {
		this->_modelLoader.setOpenVINOCPUParams(threads, num_streams);
	}


	/*
	* 设置CUDA推理参数
	*@deviceId 设置CUDA设备ID
	*/
	void setCUDAParams(unsigned short deviceId) {
		this->_modelLoader.setCUDAParams(deviceId);
	}



	//设置缩放系数
	void setScaleFactor(float scalefactor) {
		this->_normalizer.setScaleFactor(scalefactor);
	}

	//设置填充颜色
	void setFillColor(const cv::Scalar& fillColor) {
		this->_normalizer.setFillColor(fillColor);
	}

	//设置是否交换BGR通道
	void setSwapRB(bool swapRB) {
		this->_normalizer.setSwapRB(swapRB);
	}

};

using yolov8OnnxDetector = Yolov8OnnxDPImpl<DPDetector< OnnxLoader, typename SimpleLetterBoxConfig::Normalizer, OnnxRunner, typename SimpleLetterBoxConfig::DetectParser>>;

using yolov8OnnxSegmenter = Yolov8OnnxDPImpl<DPSegmentor< OnnxLoader, typename SimpleLetterBoxConfig::Normalizer, OnnxRunner, typename SimpleLetterBoxConfig::SegmentParser>>;

//不建议使用opencv::dnn::net推理，建议使用onnxruntime推理
using yolov8CVDNNCPUDetector= DPDetector< CVDnnLoaderCPU, typename SimpleLetterBoxConfig::Normalizer, CVDNNRunner, typename SimpleLetterBoxConfig::DetectParser>;
//不建议使用opencv::dnn::net推理，建议使用onnxruntime推理
using yolov8CVDNNSCPUegmenter= DPSegmentor< CVDnnLoaderCPU, typename SimpleLetterBoxConfig::Normalizer, CVDNNRunner, typename SimpleLetterBoxConfig::SegmentParser>;