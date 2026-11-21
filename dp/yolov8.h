#pragma once

#include "dp.h"

#include "letterbox.h"

#include "onnx.h"
#include "policy.h"


NAMESPACE_DP_BEGIN

class Yolov8OutputDecoder {
public:
	bool decode(const Tensor& output,
				int batchIndex,
				int classNum,
				const std::vector<float>& scoreThresholds,
				std::vector<std::vector<BoxCandidate>>& candidates,
				const cv::Size& boxBounds = cv::Size(),
				int maskCoefficientCount = 0) const {
		candidates.clear();
		if (output.info.shape.size() != 3) {
			log_error("YOLOv8预测信息张量维度应为3, 当前为{0}", output.info.shape.size());
			return false;
		}

		const int predictedClassNum = static_cast<int>(output.info.shape[1]) - 4 - maskCoefficientCount;
		if (predictedClassNum != classNum) {
			log_error("YOLOv8模型类别数与传入类别数不一致: 模型为{0}, 传入为{1}",
				predictedClassNum, classNum);
			return false;
		}

		const int predictionCount = static_cast<int>(output.info.shape[2]);
		candidates.resize(classNum);

		for (int predictionIndex = 0; predictionIndex < predictionCount; ++predictionIndex) {
			const float cx = *output.tensor.ptr<float>(batchIndex, 0, predictionIndex);
			const float cy = *output.tensor.ptr<float>(batchIndex, 1, predictionIndex);
			const float width = *output.tensor.ptr<float>(batchIndex, 2, predictionIndex);
			const float height = *output.tensor.ptr<float>(batchIndex, 3, predictionIndex);
			cv::Rect box = rect(cx, cy, width, height);
			if (boxBounds.width > 0 && boxBounds.height > 0) {
				box = rectValidate(box, boxBounds);
				if (box.empty()) {
					continue;
				}
			}

			for (int classIndex = 0; classIndex < classNum; ++classIndex) {
				const float score = *output.tensor.ptr<float>(batchIndex, 4 + classIndex, predictionIndex);
				if (score >= scoreThresholds[classIndex]) {
					candidates[classIndex].push_back({ box, score, predictionIndex });
				}
			}
		}

		return true;
	}
};

template<typename _ParseImpl, typename _OutputPolicy = NmsPolicy>
class Yolov8DetectLetterBoxResultParser : public ParserBase<
	Yolov8DetectLetterBoxResultParser<_ParseImpl, _OutputPolicy>, _ParseImpl, DetectResArray> {
public:
	DetectResArray parse(const std::vector<Tensor>& outputs,
						 const _ParseImpl& letterboxParseImpl,
						 int classNum,
						 const std::vector<float>& scoreThresholds) {
		if (outputs.size() != 1) {
			log_error("Yolov8DetectLetterBoxResultParser接受输出张量数量应为1, 当前为{0}", outputs.size());
			return DetectResArray();
		}

		const auto& output = outputs[0];
		if (output.info.shape.size() != 3) {
			log_error("YOLOv8预测信息张量维度应为3, 当前为{0}", output.info.shape.size());
			return DetectResArray();
		}

		const int batch = static_cast<int>(output.info.shape[0]);
		DetectResArray results(batch);
		Yolov8OutputDecoder decoder;
		_OutputPolicy outputPolicy;

		for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
			std::vector<std::vector<BoxCandidate>> candidatesByClass;
			if (!decoder.decode(
				output, batchIndex, classNum, scoreThresholds, candidatesByClass)) {
				return DetectResArray();
			}
			ResultBatch<DetectRes> classResults(classNum);

			for (int classIndex = 0; classIndex < classNum; ++classIndex) {
				const float iouThreshold = classIndex < static_cast<int>(iouThresholds.size())
					? iouThresholds[classIndex]
					: DP_DEFAULT_IOU_THRESHOLD;
				auto groups = outputPolicy.apply(
					candidatesByClass[classIndex],
					scoreThresholds[classIndex],
					iouThreshold);

				for (const auto& group : groups) {
					classResults[classIndex].push_back({
						letterboxParseImpl[batchIndex].enRect(group.box), group.score });
				}
			}

			results[batchIndex] = std::move(classResults);
		}

		return results;
	}

	std::vector<float> iouThresholds;
};

template<typename _ParseImpl, typename _OutputPolicy = NmsPolicy>
class Yolov8SegmentLetterBoxResultParser : public ParserBase<
	Yolov8SegmentLetterBoxResultParser<_ParseImpl, _OutputPolicy>, _ParseImpl, SegmentResArray> {
public:
	SegmentResArray parse(const std::vector<Tensor>& outputs,
						  const _ParseImpl& letterboxParseImpl,
						  int classNum,
						  const std::vector<float>& scoreThresholds) {
		if (outputs.size() != 2) {
			log_error("Yolov8SegmentLetterBoxResultParser接受输出张量数量应为2, 当前为{0}", outputs.size());
			return SegmentResArray();
		}

		const auto& detections = outputs[0];
		const auto& protos = outputs[1];
		if (detections.info.shape.size() != 3) {
			log_error("YOLOv8预测信息张量维度应为3, 当前为{0}", detections.info.shape.size());
			return SegmentResArray();
		}
		if (protos.info.shape.size() != 4) {
			log_error("YOLOv8原型掩膜张量维度应为4, 当前为{0}", protos.info.shape.size());
			return SegmentResArray();
		}

		const int batch = static_cast<int>(detections.info.shape[0]);
		const int predictionInfo = static_cast<int>(detections.info.shape[1]);
		const int maskChannels = static_cast<int>(protos.info.shape[1]);
		const int maskHeight = static_cast<int>(protos.info.shape[2]);
		const int maskWidth = static_cast<int>(protos.info.shape[3]);
		const std::vector<int> protoShape{ 1, maskChannels, maskHeight, maskWidth };

		SegmentResArray results;
		results.reserve(batch);
		Yolov8OutputDecoder decoder;
		_OutputPolicy outputPolicy;
		OrMatPolicy orMatPolicy;

		for (int batchIndex = 0; batchIndex < batch; ++batchIndex) {
			const auto& letterbox = letterboxParseImpl[batchIndex];
			const cv::Size inputSize = letterbox.targetSize();
			std::vector<std::vector<BoxCandidate>> candidatesByClass;
			if (!decoder.decode(
				detections,
				batchIndex,
				classNum,
				scoreThresholds,
				candidatesByClass,
				inputSize,
				maskChannels)) {
				return SegmentResArray();
			}
			std::vector<std::vector<BoxCandidateGroup>> groupsByClass(classNum);

			for (int classIndex = 0; classIndex < classNum; ++classIndex) {
				const float iouThreshold = classIndex < static_cast<int>(iouThresholds.size())
					? iouThresholds[classIndex]
					: DP_DEFAULT_IOU_THRESHOLD;
				groupsByClass[classIndex] = outputPolicy.apply(
					candidatesByClass[classIndex],
					scoreThresholds[classIndex],
					iouThreshold);
			}

			bool hasSelectedCandidates = false;
			for (const auto& classGroups : groupsByClass) {
				hasSelectedCandidates = hasSelectedCandidates || !classGroups.empty();
			}

			ResultBatch<SegmentRes> classResults(classNum);
			if (hasSelectedCandidates) {
				cv::Mat maskProtos(
					protoShape,
					CV_32F,
					const_cast<float*>(protos.tensor.ptr<float>(batchIndex)));
				const auto& letterboxParams = letterbox.params();

				for (int classIndex = 0; classIndex < classNum; ++classIndex) {
					for (const auto& group : groupsByClass[classIndex]) {
						const cv::Rect groupBox = letterbox.enRect(group.box);
						if (groupBox.empty()) {
							continue;
						}

						std::vector<BoxMat> memberMasks;
						memberMasks.reserve(group.members.size());
						for (const auto& member : group.members) {
							const cv::Rect memberBox = letterbox.enRect(member.box);
							if (memberBox.empty()) {
								continue;
							}

							BoxMat mask = decodeRawMask(
								detections,
								batchIndex,
								member.predictionIndex,
								classNum,
								predictionInfo,
								maskProtos,
								maskChannels,
								maskHeight,
								maskWidth,
								inputSize,
								letterboxParams,
								memberBox);
							memberMasks.push_back(std::move(mask));
						}

						if (memberMasks.empty()) {
							continue;
						}

						BoxMat rawMask;
						if (!orMatPolicy.apply(memberMasks, rawMask)) {
							log_error("NMM原型掩膜合并失败：成员掩膜类型或尺寸不一致");
							return SegmentResArray();
						}

						BoxMat finalMask = decodeMask(
							rawMask,
							maskHeight,
							maskWidth,
							inputSize,
							letterbox);
						if (finalMask.mat.empty() || finalMask.box.empty()) {
							continue;
						}

						classResults[classIndex].push_back({
							finalMask.box, group.score, std::move(finalMask.mat) });
					}
				}
			}

			results.push_back(std::move(classResults));
		}

		return results;
	}

	std::vector<float> iouThresholds;

private:
	static BoxMat decodeRawMask(const Tensor& detections,
								 int batchIndex,
								 int predictionIndex,
								 int classNum,
								 int predictionInfo,
								 const cv::Mat& maskProtos,
								 int maskChannels,
								 int maskHeight,
								 int maskWidth,
								 const cv::Size& inputSize,
								 const cv::Vec4d& letterboxParams,
								 const cv::Rect& originalBox) {
		const int netWidth = inputSize.width;
		const int netHeight = inputSize.height;
		int rangeX = static_cast<int>(std::floor(
			(originalBox.x * letterboxParams[0] + letterboxParams[2]) / netWidth * maskWidth));
		int rangeY = static_cast<int>(std::floor(
			(originalBox.y * letterboxParams[1] + letterboxParams[3]) / netHeight * maskHeight));
		int rangeWidth = static_cast<int>(std::ceil(
			((originalBox.x + originalBox.width) * letterboxParams[0] + letterboxParams[2])
			/ netWidth * maskWidth)) - rangeX;
		int rangeHeight = static_cast<int>(std::ceil(
			((originalBox.y + originalBox.height) * letterboxParams[1] + letterboxParams[3])
			/ netHeight * maskHeight)) - rangeY;

		rangeWidth = std::max(rangeWidth, 1);
		rangeHeight = std::max(rangeHeight, 1);
		if (rangeX + rangeWidth > maskWidth) {
			if (maskWidth - rangeX > 0) {
				rangeWidth = maskWidth - rangeX;
			} else {
				rangeX -= 1;
			}
		}
		if (rangeY + rangeHeight > maskHeight) {
			if (maskHeight - rangeY > 0) {
				rangeHeight = maskHeight - rangeY;
			} else {
				rangeY -= 1;
			}
		}

		std::vector<cv::Range> protoRanges{
			cv::Range(0, 1),
			cv::Range::all(),
			cv::Range(rangeY, rangeY + rangeHeight),
			cv::Range(rangeX, rangeX + rangeWidth)
		};
		cv::Mat croppedProtos = maskProtos(protoRanges).clone();
		croppedProtos = croppedProtos.reshape(0, { maskChannels, rangeWidth * rangeHeight });

		std::vector<cv::Range> coefficientRanges{
			cv::Range(batchIndex, batchIndex + 1),
			cv::Range(4 + classNum, predictionInfo),
			cv::Range(predictionIndex, predictionIndex + 1)
		};
		cv::Mat coefficients = detections.tensor(coefficientRanges).clone();
		coefficients = coefficients.reshape(0, { 1, predictionInfo - (4 + classNum) });

		cv::Mat maskFeatures = coefficients * croppedProtos;
		maskFeatures = maskFeatures.reshape(0, rangeHeight);
		cv::Mat probabilities;
		cv::exp(-maskFeatures, probabilities);
		probabilities = 1.0 / (1.0 + probabilities);
		cv::Mat rawMask;
		probabilities.convertTo(rawMask, CV_8UC1, 255);
		return { cv::Rect(rangeX, rangeY, rangeWidth, rangeHeight), std::move(rawMask) };
	}

	template<typename _LetterBox>
	static BoxMat decodeMask(const BoxMat& rawMask,
							  int maskHeight,
							  int maskWidth,
							  const cv::Size& inputSize,
							  const _LetterBox& letterbox) {
		const cv::Rect inputBox = scaleRect(rawMask.box, cv::Size(maskWidth, maskHeight), inputSize);
		const cv::Rect originalBox = letterbox.enRect(inputBox);
		if (originalBox.empty()) {
			return {};
		}

		cv::Mat originalMask;
		cv::resize(rawMask.mat, originalMask, originalBox.size(), 0, 0, cv::INTER_LINEAR);
		return { originalBox, std::move(originalMask) };
	}
};



template<bool keepScaleRatio = true,
		 bool scaleUp = false,
		 bool autoShape = false,
		 int stride = 32,
		 bool centerAnchor = true,
		 int index = 0,
		 typename _OutputPolicy = NmsPolicy>
class LetterBoxConfig {
public:
	using Box = LetterBox<keepScaleRatio, scaleUp, autoShape, stride, centerAnchor>;
	using Preprocessor = LetterBoxPreprocessor<Box, index>;
	using DetectParser = Yolov8DetectLetterBoxResultParser<LetterBoxParseImpl<Box>, _OutputPolicy>;
	using SegmentParser = Yolov8SegmentLetterBoxResultParser<LetterBoxParseImpl<Box>, _OutputPolicy>;
};

using SimpleLetterBoxConfig = LetterBoxConfig<true, false, false, 32, true, 0>;
using SimpleNmmLetterBoxConfig = LetterBoxConfig<true, false, false, 32, true, 0, NmmPolicy>;

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

	//设置 IoU 阈值
	void setIouThreshold(const std::vector<float>& iouThresholds) {
		this->_parser.iouThresholds = iouThresholds;
	}

};

using yolov8OnnxDetector = Yolov8OnnxDPImpl<DPDetector< OnnxLoader, typename SimpleLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleLetterBoxConfig::DetectParser>>;

using yolov8OnnxSegmenter = Yolov8OnnxDPImpl<DPSegmentor< OnnxLoader, typename SimpleLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleLetterBoxConfig::SegmentParser>>;

using yolov8OnnxNmmDetector = Yolov8OnnxDPImpl<DPDetector< OnnxLoader, typename SimpleNmmLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleNmmLetterBoxConfig::DetectParser>>;

using yolov8OnnxNmmSegmenter = Yolov8OnnxDPImpl<DPSegmentor< OnnxLoader, typename SimpleNmmLetterBoxConfig::Preprocessor, OnnxRunner, typename SimpleNmmLetterBoxConfig::SegmentParser>>;


NAMESPACE_DP_END