#include "yolov8.h"

#include <thread>


DetectResArray Yolov8DetectResultParser::operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize,
	std::vector<std::vector<int>> outputSizes, int classNum, std::vector<float>& socreThreshs, std::vector<float>& nmsThreshs)
{
	DetectResArray resArr(classNum);

	//只有一个输出
	cv::Mat& outputMat = outputs.at(0);
	std::vector<int>& outputSize = outputSizes.at(0);

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

		//对应原图矩形框大小
		cv::Rect box = oriRect(oriSize, inputSize, cx, cy, w, h);

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
			resArr[i].push_back(DetectRes(boxs[i][index], scores[i][index]));
		}
	}

	return resArr;
}



SegmentResArray Yolov8SegmentResultParser::operator()(std::vector<cv::Mat>& outputs, cv::Size oriSize, cv::Size inputSize,
	 std::vector<std::vector<int>> outputSizes,	int classNum, std::vector<float>& socreThreshs, std::vector<float>& nmsThreshs)
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

	for(int i=0; i<classNum; ++i) {
		for(int j=0; j<classIndexs[i].size(); ++j) {
			int index=classIndexs[i][j];

			/*
				提取掩膜特征图对应区域，计算分割掩膜，减少计算量
			*/
			//映射到分割特征图上的矩形框
			cv::Rect protoRect=scaleRect(outputBoxs[index], inputSize, cv::Size(seg_w, seg_h));

			std::vector<cv::Range> ranges;
			ranges.push_back(cv::Range(0, 1));
    		ranges.push_back(cv::Range::all());
			ranges.push_back(cv::Range(protoRect.y, protoRect.y + protoRect.height));
			ranges.push_back(cv::Range(protoRect.x, protoRect.x + protoRect.width));

			cv::Mat temp_mask_protos = mask_protos(ranges).clone();
			temp_mask_protos=temp_mask_protos.reshape(0, {seg_c, protoRect.height * protoRect.width});
			cv::Mat ceof(ceofs[index].t());

			cv::Mat mask = ceof * temp_mask_protos;
			mask = mask.reshape(0, protoRect.height);

			cv::Mat dest;
			cv::exp(-mask, dest);	//sigmoid函数
			dest= 1.0 / (1.0 + dest);

			//映射回原图像
			cv::Rect oriRect=scaleRect(outputBoxs[index], inputSize, oriSize);
			cv::Mat oriMask;
			cv::resize(dest, oriMask, oriRect.size());
			oriMask.convertTo(oriMask, CV_8UC1, 255);
			resArr[i].push_back(SegmentRes(oriRect,  scores[i][index], oriMask));
		}
	}

	return resArr;
}