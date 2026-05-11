#include "yolov8.h"

#include "log.h"

#include <random>

cv::Scalar randomColor(int seed) {
    // 使用 classId 作为种子，同类颜色一致
    std::mt19937 rng(seed * 1000);  // 乘以一个常数让相邻 classId 颜色差异更大
    std::uniform_int_distribution<int> dist(0, 255);
    return cv::Scalar(dist(rng), dist(rng), dist(rng));
}

/// 绘制检测结果，从begin到end（不包含end），-1表示全部
cv::Mat drawPred(const cv::Mat& img, const SegmentResArray& resArr, int begin=0, int end=-1) {
	cv::Mat resImg = img.clone();

	if(end<0)
		end = resArr.size();

	for(int i=begin; i<end; ++i) {
		for(int j=0; j<resArr[i].size(); ++j) {
			auto color = randomColor(i);

			auto box= resArr[i][j].box;
			cv::Mat roi = resImg(box);

			auto roiMask = resArr[i][j].mask;

			//设置掩膜区域像素为 color
			roi.setTo(color, roiMask);
		}
	}

	return resImg;
}


void testDetector() {
	const char* modelPath = R"(C:\Users\qiang\runs\detect\runs\yolov8s_singleclass_onebox_bbox_3g7\weights\best.onnx)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\document\20251026_131759_465_155.jpg)";

	//yolov8OnnxCPUDetector<> detector(1, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	yolov8CVDNNCPUDetector<> detector(1, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	detector._modelLoader.setInputSize(1, 3, 512, 512);

	detector.setNormalizeParam(1.0 / 255.0);

	detector.loadModel(modelPath);
	
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	auto resArr = detector.run(img);
	auto res = resArr.at(0);

	for (int i = 0; i < res.size(); ++i) {
		cv::rectangle(img, res[i].box, cv::Scalar(0, 0, 255), 10);
	}
	
	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", img);
}

void testSegmenter() {
	const char* modelPath = R"(D:\gw\deeplearning\yolo\yolo_dataset\bamboo\segment\512train\output\weights\best.onnx)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\document\20251026_131759_465_155.jpg)";

	//yolov8OnnxCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	yolov8CVDNNCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	segmenter._modelLoader.setInputSize(1, 3, 640, 640);

	segmenter.setNormalizeParam(1.0 / 255.0);
	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	auto resArr = segmenter.run(img);

	cv::Mat resImg = drawPred(img, resArr);

	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", resImg);
}

int main()
{
	logInit(Log_Level::info);

	//testDetector();
	testSegmenter();

	cv::waitKey();

	return 0;
}