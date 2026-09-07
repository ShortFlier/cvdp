#include "dputility.h"

cv::Size getImageInputSize(const std::vector<TensorInfo>& inputTensorInfos, int index){
	if(index>=0)
		return cv::Size(inputTensorInfos[index].shape[3], inputTensorInfos[index].shape[2]);

	// 自动搜索第一个4维的输入张量
	for(size_t i=0;i<inputTensorInfos.size();i++){
		if(inputTensorInfos[i].shape.size()==4)
			return cv::Size(inputTensorInfos[i].shape[3], inputTensorInfos[i].shape[2]);
	}
	
	return cv::Size();
}

cv::Rect rect(double cx, double cy, double w, double h) {
	int ltx= static_cast<int>(std::round(cx - w / 2));
	int lty= static_cast<int>(std::round(cy - h / 2));

	int rbx= static_cast<int>(std::round(cx + w / 2));
	int rby= static_cast<int>(std::round(cy + h / 2));


	return cv::Rect(ltx, lty, rbx - ltx, rby - lty);
}

//将矩形框从一个尺寸缩放到另一个尺寸
cv::Rect scaleRect(const cv::Rect& box, const cv::Size& fromSize, const cv::Size& toSize) {
	double x_scale = static_cast<double>(toSize.width) / fromSize.width;
	double y_scale = static_cast<double>(toSize.height) / fromSize.height;

	int ltx = static_cast<int>(std::round(box.x * x_scale));
	int lty = static_cast<int>(std::round(box.y * y_scale));
	int rbx = static_cast<int>(std::round((box.x + box.width) * x_scale));
	int rby = static_cast<int>(std::round((box.y + box.height) * y_scale));

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

