#include "yolov8.h"

#include "log.h"

#include <random>
#include <algorithm>
#include <numeric>

cv::Scalar_<uchar> randomColor(int seed) {
    // 使用 classId 作为种子，同类颜色一致
    // std::mt19937 rng(seed * 1000);  // 乘以一个常数让相邻 classId 颜色差异更大
    // std::uniform_int_distribution<int> dist(0, 255);
    // return cv::Scalar(cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)), cv::saturate_cast<uchar>(dist(rng)));

	cv::Scalar rgb[]={cv::Scalar(0, 0, 255), cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0)};
	return rgb[seed%3];
}


/// 绘制检测结果
void showDetectRes(const std::vector<cv::Mat>& img, const DetectResArray& resArr) {

	log_info("detect result batch: {0}", resArr.size());
	for(int i=0; i<img.size(); ++i){
		std::string windowName = "det_batch_" + std::to_string(i);
		
		cv::Mat resImg = img[i].clone();

		auto& res = resArr[i];
		//每个类别
		for(int j=0; j<res.size(); ++j) {
			auto color = randomColor(j);

			for(int k=0; k<res[j].boxs.size(); ++k) {
				cv::rectangle(resImg, res[j].boxs[k], color, 2);
			}

		}

		cv::namedWindow(windowName, cv::WINDOW_NORMAL);
		cv::imshow(windowName, resImg);
	}
}

/// 绘制分割结果
void showSegmentRes(const std::vector<cv::Mat>& img, const SegmentResArray& resArr) {

	for(int i=0; i<img.size(); ++i){
		std::string windowName = "seg_batch_" + std::to_string(i);
		
		cv::Mat resImg = img[i].clone();

		auto& res = resArr[i];
		//每个类别
		for(int j=0; j<res.size(); ++j) {
			auto color = randomColor(j);

			for(int k=0; k<res[j].masks.size(); ++k) {
				cv::Mat mask = res[j].masks[k];
				int roff=res[j].boxs[k].y;
				int coff=res[j].boxs[k].x;
				for(int r=0; r<mask.rows; ++r) {
					for(int c=0; c<mask.cols; ++c) {
						if(mask.at<uchar>(r, c) > 130) {
							cv::Vec3b& v = resImg.at<cv::Vec3b>(r+roff, c+coff);
							v = cv::Vec3b(v[0] | color[0], v[1] | color[1], v[2] | color[2]);
						}
					}
				}
				cv::rectangle(resImg, res[j].boxs[k], color, 2);
			}

		}

		cv::namedWindow(windowName, cv::WINDOW_NORMAL);
		cv::imshow(windowName, resImg);
	}
}


//耗时检测
template<typename Dp>
void elapsedTime(Dp& dp, const std::vector<cv::Mat>& imgs, int iterations) {
	std::vector<int> mses;
	for(int i=0; i<iterations; ++i){
		auto start = std::chrono::high_resolution_clock::now();
		dp.run(imgs);
		auto end = std::chrono::high_resolution_clock::now();
		int elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
		mses.push_back(elapsed);
	}
	int max, min, avg;
	max = *std::max_element(mses.begin(), mses.end());
	min = *std::min_element(mses.begin(), mses.end());
	avg = std::accumulate(mses.begin(), mses.end(), 0) / mses.size();
	log_debug("Elapsed times (ms): {}", fmt::join(mses, ", "));
	log_info("Elapsed time: max = {} ms, min = {} ms, avg = {} ms", max, min, avg);
}

//const char* imgPath="test/wtest.png";
const char* imgPath="test/test.jpg";

// const char* detectModelPath="model/detect.onnx";
// const int detectClassNum=1;

const char* detectModelPath="model/wdetect4cls.onnx";
const int detectClassNum=4;

const char* segmentModelPath="model/segment.onnx";
//const char* segmentModelPath="model/dynseg.onnx";
const int segmentClassNum=2;



void testDetectorCPU() {
	const char* modelPath = detectModelPath;

	yolov8OnnxDetector detector(detectClassNum, std::vector<float>({ 0.25f }), std::vector<float>({ 0.45f }));
	//设置为使用CPU推理
	detector._modelLoader.setDeviceType(OnnxLoader::DeviceType::CPU);
	//detector.setCPUParams(2, 0);

	detector.loadModel(modelPath);
	
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
	auto resArr = detector.run(std::vector<cv::Mat>({img}));

	showDetectRes(std::vector<cv::Mat>({img}), resArr);

	//elapsedTime(detector, std::vector<cv::Mat>({img}), 10);
}

void testSegmenterCPU() {
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum);
	//设置为使用CPU推理
	segmenter.setDeviceType(OnnxLoader::DeviceType::CPU);

	segmenter.setDynamicInputSize(cv::Size(640, 640)); 

	//segmenter.setCPUParams(2,0);

	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	auto resArr = segmenter.run(std::vector<cv::Mat>({img}));

	showSegmentRes(std::vector<cv::Mat>({img}), resArr);

	
}

void testDetectorCUDA() {
	
	const char* modelPath = detectModelPath;

	yolov8OnnxDetector detector(detectClassNum, std::vector<float>({ 0.25f }), std::vector<float>({ 0.45f }));
	detector.loadModel(modelPath);
		cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

		//运行1次预热
		auto resArr = detector.run(std::vector<cv::Mat>({img}));

		resArr = detector.run(std::vector<cv::Mat>({img}));
		showDetectRes(std::vector<cv::Mat>({img}), resArr);
}

void testSegmenterCUDA() {
	
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum, std::vector<float>({ 0.5f, 0.5f }), std::vector<float>({ 0.5f, 0.5f }));


	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	
	//预热一次
	segmenter.run(std::vector<cv::Mat>({img}));

	auto resArr = segmenter.run(std::vector<cv::Mat>({img}));

	showSegmentRes(std::vector<cv::Mat>({img}), resArr);
}


void testSegmenterOpenVINO() {
	
	const char* modelPath = segmentModelPath;
	yolov8OnnxSegmenter segmenter(segmentClassNum, std::vector<float>({ 0.5f, 0.5f }), std::vector<float>({ 0.5f, 0.5f }));
	segmenter.setDeviceType(OnnxLoader::DeviceType::OpenVINO_CPU);
	segmenter.setOpenVINOCPUParams(0, 1);

	segmenter.loadModel(modelPath);
	cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);

	
	//预热一次
	segmenter.run(std::vector<cv::Mat>({img}));
	log_info("预热完成");

	auto resArr = segmenter.run(std::vector<cv::Mat>({img}));
	showSegmentRes(std::vector<cv::Mat>({img}), resArr);

}

//对比单输入与动态输入运行速度
void compareDynAndSta(){
	const char* imgDir="test/imgs";

	const char* dynModel="model/dynseg.onnx";
	const char* staModel="model/segment.onnx";

	const int classNum = 2;
	const int warmupRuns = 1;
	const int measureRuns = 5;

	//获取所有图片
	std::vector<cv::Mat> imgs;
	std::vector<cv::String> imagePaths;
	cv::glob(std::string(imgDir) + "/*", imagePaths, false);
	for (const auto& imagePath : imagePaths) {
		cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
		if (image.empty()) {
			log_warn("无法读取图片，跳过：{}", imagePath);
			continue;
		}
		imgs.push_back(std::move(image));
	}

	if (imgs.empty()) {
		log_error("目录中没有可用于对比的图片：{}", imgDir);
		return;
	}

	//初始化模型
	yolov8OnnxSegmenter staticSegmenter(classNum);
	yolov8OnnxSegmenter dynamicSegmenter(classNum);
	dynamicSegmenter.setDynamicInputSize(cv::Size(640, 640));
	staticSegmenter.loadModel(staModel);
	dynamicSegmenter.loadModel(dynModel);

	//预热，避免把首次运行的初始化开销计入结果
	for (int i = 0; i < warmupRuns; ++i) {
		for (const auto& image : imgs) {
			staticSegmenter.run(std::vector<cv::Mat>{image});
		}
		dynamicSegmenter.run(imgs);
	}

	const auto measure = [&imgs, measureRuns](auto& segmenter, bool batchInput) {
		const auto start = std::chrono::steady_clock::now();
		for (int run = 0; run < measureRuns; ++run) {
			if (batchInput) {
				segmenter.run(imgs);
			} else {
				for (const auto& image : imgs) {
					segmenter.run(std::vector<cv::Mat>{image});
				}
			}
		}
		const auto end = std::chrono::steady_clock::now();
		return std::chrono::duration<double, std::milli>(end - start).count();
	};

	const double staticElapsedMs = measure(staticSegmenter, false);
	const double dynamicElapsedMs = measure(dynamicSegmenter, true);
	const double staticAverageMs = staticElapsedMs / measureRuns;
	const double dynamicAverageMs = dynamicElapsedMs / measureRuns;

	//打印时间对比
	log_info("图片数量：{}，测量次数：{}", imgs.size(), measureRuns);
	log_info("静态模型单张输入：总耗时 {:.3f} ms，平均每批 {:.3f} ms",
		staticElapsedMs, staticAverageMs);
	log_info("动态模型批量输入：总耗时 {:.3f} ms，平均每批 {:.3f} ms",
		dynamicElapsedMs, dynamicAverageMs);
	if (dynamicElapsedMs > 0.0) {
		log_info("动态批量相对静态逐张加速：{:.2f}x",
			staticElapsedMs / dynamicElapsedMs);
	}
}

int main()
{
	SetConsoleOutputCP(CP_UTF8);

	logInit(Log_Level::debug);

	//testDetectorCPU();
	//testSegmenterCPU();
	//testDetectorCUDA();
	//testSegmenterCUDA();
	//testSegmenterOpenVINO();

	compareDynAndSta();

	cv::waitKey();

	return 0;
}