#include "yolov8.h"

#include "log.h"

#include <random>

cv::Scalar_<uchar> randomColor(int seed) {
    // 使用 classId 作为种子，同类颜色一致
    // std::mt19937 rng(seed * 1000);  // 乘以一个常数让相邻 classId 颜色差异更大
    // std::uniform_int_distribution<int> dist(0, 255);
    // return cv::Scalar(cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)));

	cv::Scalar rgb[]={cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
	return rgb[seed%3];
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

	cv::imwrite("res.jpg", resImg);

	return resImg;
}

//const char* imgPath="test/wtest.png";
const char* imgPath="test/test.jpg";

const char* detectModelPath="model/detect.onnx";
const int detectClassNum=1;

// const char* detectModelPath="model/wdetect4cls.onnx";
// //const char* detectModelPath="model/wpyoloe.onnx";
// const int detectClassNum=4;

const char* segmentModelPath="model/segment.onnx";
const int segmentClassNum=2;



void testDetectorCPU() {
	const char* modelPath = detectModelPath;

	yolov8OnnxDetector detector(detectClassNum, std::vector<float>({ 0.25 }), std::vector<float>({ 0.45 }));
	//设置为使用CPU推理
	detector.setDeviceType(OnnxLoader::DeviceType::CPU);
	//detector.setCPUParams(2, 0);

	detector.loadModel(modelPath);
	
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	auto resArr = detector.run(img);

	for(int j=0; j<resArr.size(); ++j) {
		auto res = resArr[j];
		log_info("class id: {0}, box number: {1}", j, res.size());
		for (int i = 0; i < res.size(); ++i) {
			cv::rectangle(img, res[i].box, randomColor(j), 6);
		}
	}

	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", img);
}

void testSegmenterCPU() {
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum);
	//设置为使用CPU推理
	segmenter.setDeviceType(OnnxLoader::DeviceType::CPU);
	//segmenter.setCPUParams(2,0);

	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	do{
		auto resArr = segmenter.run(img);

		cv::Mat resImg = drawPred(img, resArr);

		cv::namedWindow("res", cv::WINDOW_NORMAL);
		cv::imshow("res", resImg);
	}while(cv::waitKey() != 27); // 按ESC键退出
}

void testDetectorCUDA() {
	
	const char* modelPath = detectModelPath;

	yolov8OnnxDetector detector(detectClassNum, std::vector<float>({ 0.25f }), std::vector<float>({ 0.45f }));
	detector.loadModel(modelPath);
	for(int i=0; i<2; ++i){


		cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
		auto resArr = detector.run(img);

		if(i<1){//运行1次预热
			continue;
		}

		for(int j=0; j<resArr.size(); ++j) {
			auto res = resArr[j];
			log_info("class id: {0}, box number: {1}", j, res.size());
			for (int i = 0; i < res.size(); ++i) {
				cv::rectangle(img, res[i].box, randomColor(j), 6);
			}
		}

		cv::namedWindow("onnx_loader_test", cv::WINDOW_NORMAL);
		cv::imshow("onnx_loader_test", img);
	}
}

void testSegmenterCUDA() {
	
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum);


	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	
	//预热一次
	segmenter.run(img);

	auto resArr = segmenter.run(img);

	cv::Mat resImg = drawPred(img, resArr);

	cv::namedWindow("res", cv::WINDOW_NORMAL);
	cv::imshow("res", resImg);
}


void testSegmenterOpenVINO() {
	
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum, std::vector<float>({ 0.5f, 0.5f }), std::vector<float>({ 0.5f, 0.5f }));
	segmenter.setDeviceType(OnnxLoader::DeviceType::OpenVINO_CPU);
	segmenter.setOpenVINOCPUParams(0, 1);

	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	
	//预热一次
	segmenter.run(img);
	log_info("预热完成");

	while(true){
		auto resArr = segmenter.run(img);

		cv::Mat resImg = drawPred(img, resArr);

		cv::destroyAllWindows();
		cv::namedWindow("res", cv::WINDOW_NORMAL);
		cv::imshow("res", resImg);

		log_info("按任意键继续，按ESC退出");
		int key = cv::waitKey();
		if (key == 27) { // ESC key
			break;
		}
	}
}

int main()
{
	SetConsoleOutputCP(CP_UTF8);

	logInit(Log_Level::info);

	//testDetectorCPU();
	testSegmenterCPU();
	//testDetectorCUDA();
	//testSegmenterCUDA();
	//testSegmenterOpenVINO();

	cv::waitKey();

	return 0;
}