#include "yolov8.h"

#include "log.h"

#include <random>

cv::Scalar_<uchar> randomColor(int seed) {
    // // 使用 classId 作为种子，同类颜色一致
    // std::mt19937 rng(seed * 1000);  // 乘以一个常数让相邻 classId 颜色差异更大
    // std::uniform_int_distribution<int> dist(0, 255);
    // return cv::Scalar(cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)));
	cv::Scalar_<uchar> color;
	switch(seed){
		case 0: color=cv::Scalar_<uchar>(255, 0, 0); break;
		case 1: color=cv::Scalar_<uchar>(0, 255, 0); break;
		case 2: color=cv::Scalar_<uchar>(0, 0, 255); break;
		default: color=cv::Scalar_<uchar>(255, 255, 0); break;
	}
	return color;
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

			cv::rectangle(resImg, box, color, 2);
			//设置掩膜区域像素为 color
			for (int r = 0; r < roi.rows; ++r) {
				for (int c = 0; c < roi.cols; ++c) {
					if (roiMask.at<uchar>(r, c)>130) {
						cv::Vec3b& v = roi.at< cv::Vec3b>(r, c);
						v = cv::Vec3b(v[0] | color[0], v[1] | color[1], v[2] | color[2]);
					}
				}
			}
		}
	}

	return resImg;
}


void testDetector() {
	const char* modelPath = R"(C:\Users\qiang\runs\detect\runs\yolov8s_singleclass_onebox_bbox_3g7\weights\best.onnx)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\document\20251026_131759_465_155.jpg)";

	yolov8OnnxDetector detector(1, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	//设置为使用CPU推理
	detector._modelLoader.setUsingGPU(false);
	// yolov8CVDNNDetector<> detector(1, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	// detector._modelLoader.setInputSize(1, 3, 512, 512);

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
	const char* modelPath = R"(D:\gw\deeplearning\yolo\yolo_dataset\bamboo\segment\512train\output\weights\best640x640.onnx)";
	//const char* imgPath = R"(D:\gw\deeplearning\yolo\yolo_dataset\bamboo\segment\512train\train\images\1.jpg)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\test\20251026_142007_456_542.jpg)";

	//yolov8OnnxCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	yolov8OnnxSegmenter segmenter(2);
	//设置为使用CPU推理
	segmenter._modelLoader.setUsingGPU(false);
	// yolov8CVDNNCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	// segmenter._modelLoader.setInputSize(1, 3, 640, 640);

	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	auto resArr = segmenter.run(img);

	cv::Mat resImg = drawPred(img, resArr);

	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", resImg);
}

void testOnnxLoaderGpuFallbackDetect() {
	
	const char* modelPath = R"(C:\Users\qiang\runs\detect\runs\yolov8s_singleclass_onebox_bbox_3g7\weights\best.onnx)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\document\20251026_131759_465_155.jpg)";

	yolov8OnnxDetector detector(1, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	detector.loadModel(modelPath);
	for(int i=0; i<4; ++i){


		cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
		auto resArr = detector.run(img);

		if(i<3){//运行3次预热
			continue;
		}

		auto res = resArr.at(0);

		for (int i = 0; i < res.size(); ++i) {
			cv::rectangle(img, res[i].box, cv::Scalar(0, 255, 0), 6);
		}

		cv::namedWindow("onnx_loader_test", cv::WINDOW_NORMAL);
		cv::imshow("onnx_loader_test", img);
	}
}

void testOnnxLoaderGpuFallbackSegment() {
	
	const char* modelPath = R"(D:\gw\deeplearning\yolo\yolo_dataset\bamboo\segment\512train\output\weights\best640x640.onnx)";
	//const char* imgPath = R"(D:\gw\deeplearning\yolo\yolo_dataset\bamboo\segment\512train\train\images\1.jpg)";
	const char* imgPath = R"(C:\Users\qiang\Desktop\test\20251026_142007_456_542.jpg)";

	//yolov8OnnxCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	yolov8OnnxSegmenter segmenter(2);
	// yolov8CVDNNCPUSegmenter<> segmenter(2, std::vector<float>({ 0.25, 0.25 }), std::vector<float>({ 0.45, 0.45 }));
	// segmenter._modelLoader.setInputSize(1, 3, 640, 640);


	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	
	//预热一次
	segmenter.run(img);

	auto resArr = segmenter.run(img);

	cv::Mat resImg = drawPred(img, resArr);

	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", resImg);
}

int main()
{
	SetConsoleOutputCP(CP_UTF8);

	logInit(Log_Level::info);

	//testDetector();
	//testSegmenter();
	//testOnnxLoaderGpuFallbackDetect();
	testOnnxLoaderGpuFallbackSegment();

	cv::waitKey();

	return 0;
}