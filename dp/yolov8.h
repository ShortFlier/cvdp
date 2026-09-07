#pragma once

#include "dp.h"

#include "letterbox.h"

#include "onnx.h"




//yolov8检测模型结果解析
template<typename _ParseImpl>
class Yolov8DetectLetterBoxResultParser:public ParserBase<Yolov8DetectLetterBoxResultParser<_ParseImpl>, _ParseImpl, DetectResArray>{
public:
	Yolov8DetectLetterBoxResultParser(){}

	DetectResArray parse(const std::vector<Tensor>& outputs,
						const _ParseImpl& letterboxParseImpl,
						int classNum,
		  				const std::vector<float>& socreThreshs,
		   				const std::vector<float>& nmsThreshs);

};

template<typename _ParseImpl>
DetectResArray Yolov8DetectLetterBoxResultParser<_ParseImpl>::parse(const std::vector<Tensor>& outputs,
																		const _ParseImpl& letterboxParseImpl,
																		int classNum,
																		const std::vector<float>& socreThreshs,
																		const std::vector<float>& nmsThreshs)
{
	//只有一个输出张量
	if(outputs.size() != 1) {
		std::string errMsg = "Yolov8DetectLetterBoxResultParser接受输出张量数量应为1, 当前为" + std::to_string(outputs.size());
		log_error(errMsg);
		return DetectResArray();
	}

	auto& outputTensor=outputs[0];

	//获取batch
	auto& batch = outputTensor.info.shape[0];

	DetectResArray resArr(batch);

	//解析每个batch
	//yolov8输出格式[batch，类别属性，预测数]
	auto& predInfo = outputTensor.info.shape[1];
	auto& predNum = outputTensor.info.shape[2];

	auto& tensor=outputTensor.tensor;
	
	//获取每个batch的预测结果
	for (int b = 0; b < batch; ++b) {
		std::vector<std::vector<float>> scores(classNum);
		std::vector<std::vector<cv::Rect>> boxs(classNum);
		std::vector<DetectRes> detRes(classNum);

		//每个列为一个预测，包含(cx, cy, w, h, class1_score, class2_score, ...)
		for(int p = 0; p < predNum; ++p) {
			float cx = *tensor.ptr<float>(b, 0, p);
			float cy = *tensor.ptr<float>(b, 1, p);
			float w = *tensor.ptr<float>(b, 2, p);
			float h = *tensor.ptr<float>(b, 3, p);

			cv::Rect box = rect(cx, cy, w, h);

			for (int i = 0; i < classNum; ++i) {
				float score=*tensor.ptr<float>(b, 4 + i, p);
				if(score >= socreThreshs[i]){
					scores[i].push_back(score);
					boxs[i].push_back(box);
				}
			}
		}

		//nms过滤获取值
		for (int i = 0; i < classNum; ++i) {
			std::vector<int> indexs;
			cv::dnn::NMSBoxes(boxs[i], scores[i], socreThreshs[i], nmsThreshs[i], indexs);

			DetectRes det;
			for (int j = 0; j < indexs.size(); ++j) {
				int index = indexs[j];

				//对应原图矩形框大小
				cv::Rect box = letterboxParseImpl[b].enRect(boxs[i][index]);

				det.boxs.push_back(box);
				det.scores.push_back(scores[i][index]);
			}

			detRes[i] = std::move(det);
		}

		resArr[b] = std::move(detRes);
	}

	return resArr;
}






//yolov8分割模型结果解析
template<typename _ParseImpl>
class Yolov8SegmentLetterBoxResultParser:public ParserBase<Yolov8SegmentLetterBoxResultParser<_ParseImpl>, _ParseImpl, SegmentResArray>{
public:
	Yolov8SegmentLetterBoxResultParser(){}

	SegmentResArray parse(const std::vector<Tensor>& outputs,
							const _ParseImpl& letterboxParseImpl,
							int classNum,
							const std::vector<float>& socreThreshs,
							const std::vector<float>& nmsThreshs);

};


template<typename _ParseImpl>
SegmentResArray Yolov8SegmentLetterBoxResultParser<_ParseImpl>::parse(const std::vector<Tensor>& outputs,
							const _ParseImpl& letterboxParseImpl,
							int classNum,
							const std::vector<float>& socreThreshs,
							const std::vector<float>& nmsThreshs)
{
	//2个输出张量
	if(outputs.size() != 2) {
		std::string errMsg = "Yolov8SegmentLetterBoxResultParser接受输出张量数量应为2, 当前为" + std::to_string(outputs.size());
		log_error(errMsg);
		return SegmentResArray();
	}

	//检测与系数张量，形状为[batch, 预测信息, 预测数]
	auto& output0=outputs[0];
	//原型掩膜张量,形状为[N, ceof, H_proto, W_proto]
	auto& protos=const_cast<Tensor&>(outputs[1]);
	
	//batch
	auto& batch = output0.info.shape[0];

	SegmentResArray resArr;

	auto& preNum=output0.info.shape[2];
	int preInfo = output0.info.shape[1];

	//mask_protos信息
	int seg_c=protos.info.shape[1];
	int seg_h=protos.info.shape[2];
	int seg_w=protos.info.shape[3];
	std::vector<int> protosShape{1, seg_c, seg_h, seg_w};

	// 遍历每个 batch 的输出，进行解析
	for(int b=0; b<batch; ++b) {
		
		auto& _letterBox=letterboxParseImpl[b];
		auto& inputSize = _letterBox.targetSize();
		//处理检测信息
		std::vector<std::vector<cv::Rect>> outputBoxs(classNum);
		std::vector<std::vector<float>> scores(classNum);
		std::vector<std::vector<int>> predictionIndices(classNum);

		//初步筛选
		//检测张量[batch, 预测信息, 预测数]
		//每列为[center_x, center_y, width, height, class1_score, ..., classN_score, coef1, ..., coefM]
		for(int p=0; p<preNum; ++p) {
			float cx= *output0.tensor.ptr<float>(b, 0, p);
			float cy= *output0.tensor.ptr<float>(b, 1, p);
			float w= *output0.tensor.ptr<float>(b, 2, p);
			float h= *output0.tensor.ptr<float>(b, 3, p);

			auto box=rectValidate(rect(cx, cy, w, h), inputSize);

			//保存box框及对应的分数
			for(int i=0; i<classNum; ++i) {
				float score= *output0.tensor.ptr<float>(b, 4+i, p);
				if(score >= socreThreshs[i]) {
					outputBoxs[i].push_back(box);
					scores[i].push_back(score);
					predictionIndices[i].push_back(p);
				}
			}
		}

		//NMS操作
		std::vector<std::vector<int>> classIndexs(classNum);
		for (int i = 0; i < classNum; ++i) {
			cv::dnn::NMSBoxes(outputBoxs[i], scores[i], socreThreshs.at(i), nmsThreshs.at(i), classIndexs[i]);
		}


		/*
		根据NMS结果，获取对应的掩膜特征图，生成分割掩膜
		*/

		//分离该批次的原型掩膜
		cv::Mat mask_protos(protosShape, CV_32F, protos.tensor.ptr<float>(b));

		auto& params = _letterBox.params();

		std::vector<SegmentRes> segmentResList(classNum);

		for(int i=0; i<classNum; ++i) {
			for(int j=0; j<classIndexs[i].size(); ++j) {
				int index=classIndexs[i][j];
				int predictionIndex = predictionIndices[i][index];

				/*
					提取掩膜特征图对应区域，计算分割掩膜，减少计算量
				*/
				cv::Rect oriRect = _letterBox.enRect(outputBoxs[i][index]);
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

				// 提取当前检测框对应的掩膜系数
				std::vector<cv::Range> ceofRanges;
				ceofRanges.push_back(cv::Range(b, b+1));
				ceofRanges.push_back(cv::Range(4+classNum, preInfo));
				ceofRanges.push_back(cv::Range(predictionIndex, predictionIndex+1));
				cv::Mat ceof=output0.tensor(ceofRanges).clone();

				ceof=ceof.reshape(0, {1, preInfo - (4+classNum)});

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

				segmentResList[i].scores.push_back(scores[i][index]);
				segmentResList[i].masks.push_back(oriMask);
				segmentResList[i].boxs.push_back(oriRect);

			}
		}

		resArr.push_back(std::move(segmentResList));

	}


	return resArr;
}



template<bool keepScaleRatio = true, bool scaleUp = false, bool autoShape = false, int stride = 32, bool centerAnchor = true, int index=0>
class LetterBoxConfig {
public:
	using Box = LetterBox<keepScaleRatio, scaleUp, autoShape, stride, centerAnchor>;
	using Preprocessor = LetterBoxPreprocessor<Box, index>;
	using DetectParser = Yolov8DetectLetterBoxResultParser<LetterBoxParseImpl<Box>>;
	using SegmentParser = Yolov8SegmentLetterBoxResultParser<LetterBoxParseImpl<Box>>;
};

using SimpleLetterBoxConfig = LetterBoxConfig<true, false, false, 32, true, 0>;

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

using yolov8OnnxDetector = Yolov8OnnxDPImpl<DPDetector< OnnxLoader, typename SimpleLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleLetterBoxConfig::DetectParser>>;

using yolov8OnnxSegmenter = Yolov8OnnxDPImpl<DPSegmentor< OnnxLoader, typename SimpleLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleLetterBoxConfig::SegmentParser>>;
