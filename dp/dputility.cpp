#include "dputility.h"

cv::Rect rect(double cx, double cy, double w, double h) {
	int x= static_cast<int>(std::floor(cx - w / 2));
	int y= static_cast<int>(std::floor(cy - h / 2));
	int width= static_cast<int>(std::ceil(w));
	int height= static_cast<int>(std::ceil(h));
	return cv::Rect(x, y, width, height);
}

//将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize) {
	double x_scale = static_cast<double>(toSize.width) / fromSize.width;
	double y_scale = static_cast<double>(toSize.height) / fromSize.height;

	int ltx = static_cast<int>(std::floor(box.x * x_scale));
	int lty = static_cast<int>(std::floor(box.y * y_scale));
	int rbx = static_cast<int>(std::ceil((box.x + box.width) * x_scale));
	int rby = static_cast<int>(std::ceil((box.y + box.height) * y_scale));

	int width = rbx - ltx;
	int height = rby - lty;

	return cv::Rect(ltx, lty, width, height);
}

cv::Rect rectValidate(const cv::Rect& box, const cv::Size& size){
	cv::Rect validBox = box;
	//宽高至少为1
	validBox.width = std::max(validBox.width, 1);
	validBox.height = std::max(validBox.height, 1);

	cv::Rect imgRect(0, 0, size.width, size.height);
	return validBox & imgRect; // 取交集，确保在图片范围内
}

