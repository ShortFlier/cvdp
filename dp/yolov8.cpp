#include "yolov8.h"

#include <thread>


DetectResArray Yolov8DetectResultParser::operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize,
	const std::vector<std::vector<int>>& outputSizes, int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs)
{
	DetectResArray resArr(classNum);

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
			cv::Rect box = enSimpleLetterBoxRect(oriSize, inputSize, boxs[i][index]);

			resArr[i].push_back(DetectRes(box, scores[i][index]));
		}
	}

	return resArr;
}

SegmentResArray Yolov8SegmentResultParser::operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize,
	 const std::vector<std::vector<int>>& outputSizes,	int classNum, const std::vector<float>& socreThreshs, const std::vector<float>& nmsThreshs)
{
	SegmentResArray resArr(classNum);

	/*
	outputs应该有两个输出张量
	一个是特征输出[1, 预测信息，预测数]，预测信息格式[cx, cy, w, h, class1_score, ..., ceof]
	另一个是原型掩膜特征图[1,ceof, h, w]
	*/
	if(outputs.size() != 2) {
		std::string errMsg = "Yolov8SegmentResultParser::operator() error: outputs size should be 2, but get " + std::to_string(outputs.size());
		log_error(errMsg);
		throw std::runtime_error(errMsg);
	}

	/*
		获取每个类别分数，进行NMS操作，初步筛选
	*/	
	std::vector<cv::Rect> outputBoxs;
	std::vector<std::vector<float>> scores(classNum);
	std::vector<cv::Mat> ceofs;

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

		ceofs.push_back(predMat.col(c).rowRange(4 + classNum, predMat.rows));
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

	SimpleLetterBox letterbox(oriSize, inputSize, cv::Scalar(114, 114, 114));
	cv::Vec4d params = letterbox.params();

	for(int i=0; i<classNum; ++i) {
		for(int j=0; j<classIndexs[i].size(); ++j) {
			int index=classIndexs[i][j];

			/*
				提取掩膜特征图对应区域，计算分割掩膜，减少计算量
			*/
			cv::Rect oriRect = letterbox.enRect(outputBoxs[index]);
			int net_width = inputSize.width;
			int net_height = inputSize.height;

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

			cv::Mat temp_mask_protos = mask_protos(ranges).clone();
			temp_mask_protos = temp_mask_protos.reshape(0, {seg_c, rang_w * rang_h});
			cv::Mat ceof(ceofs[index].t());

						cv::Mat mask_feature = ceof * temp_mask_protos;
			mask_feature = mask_feature.reshape(0, rang_h);

			cv::Mat dest;
			cv::exp(-mask_feature, dest);
			dest = 1.0 / (1.0 + dest);

			int left = static_cast<int>(std::floor((net_width / static_cast<double>(seg_w) * rang_x - params[2]) / params[0]));
			int top = static_cast<int>(std::floor((net_height / static_cast<double>(seg_h) * rang_y - params[3]) / params[1]));
			int width = static_cast<int>(std::ceil(net_width / static_cast<double>(seg_w) * rang_w / params[0]));
			int height = static_cast<int>(std::ceil(net_height / static_cast<double>(seg_h) * rang_h / params[1]));

			cv::Mat maskPatch;
			cv::resize(dest, maskPatch, cv::Size(width, height), cv::INTER_NEAREST);
			cv::Mat finalMask = maskPatch(oriRect - cv::Point(left, top));

			cv::Mat oriMask;
			finalMask.convertTo(oriMask, CV_8UC1, 255);
			resArr[i].push_back(SegmentRes(oriRect,  scores[i][index], oriMask));

		}
	}

	return resArr;
}


