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

