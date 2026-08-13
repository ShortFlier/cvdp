#pragma once

#include <opencv2/opencv.hpp>
#include <thread>

// 获取当前CPU线程数的一半
inline int getCPUConcurrency() {
	return static_cast<int>(std::thread::hardware_concurrency())/2;
}




// 根据中心点坐标和宽高生成矩形框
inline cv::Rect rect(double cx, double cy, double w, double h) {
	return cv::Rect(cv::Point(cx - w / 2, cy - h / 2), cv::Size(w, h));
}

// 将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize);

// 将矩形框限制在图片范围内
cv::Rect rectValidate(const cv::Rect& box, const cv::Size& size);


/*
* @param oriSize 原图片大小
* @param inputSize 模型输入图片大小
* @param cx，cy，w，h矩形框中心位置x,中心位置y，宽高，对应输入图片中的矩形框
*
* @note 将输入图片中的矩形框转为原始图片中的矩形框
*/
inline cv::Rect oriRect(cv::Size oriSize, cv::Size inputSize, float cx, float cy, float w, float h){
	return scaleRect(rect(cx, cy, w, h), inputSize, oriSize);
}

