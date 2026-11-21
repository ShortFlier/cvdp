#include "yolov8.h"

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

void check(bool condition) {
	if (!condition) {
		throw std::runtime_error("result array test failed");
	}
}

template<typename Function>
void checkOutOfRange(Function function) {
	try {
		function();
	} catch (const std::out_of_range&) {
		return;
	}
	throw std::runtime_error("expected std::out_of_range");
}

template<typename Array>
void checkAccess(Array& results) {
	check(&results.batchAt(0).classAt(0) == &results.batchClassAt(0, 0));
	const Array& constResults = results;
	check(&constResults.batchAt(0).classAt(0) == &constResults.batchClassAt(0, 0));
	checkOutOfRange([&] { results.batchAt(-1); });
	checkOutOfRange([&] { constResults.batchClassAt(2, 0); });
	checkOutOfRange([&] { results.batchAt(0).classAt(-1); });
	checkOutOfRange([&] { constResults.batchClassAt(0, 2); });
}

dp::Tensor predictions(bool segmentation) {
	const int channels = segmentation ? 7 : 6;
	const int dimensions[] = { 2, channels, 3 };
	dp::Tensor result{ { "predictions", { 2, channels, 3 } },
		cv::Mat(3, dimensions, CV_32F, cv::Scalar(0)) };
	for (int batch = 0; batch < 2; ++batch) {
		for (int index = 0; index < 3; ++index) {
			*result.tensor.ptr<float>(batch, 0, index) = 4.0f + index * 4.0f;
			*result.tensor.ptr<float>(batch, 1, index) = 4.0f;
			*result.tensor.ptr<float>(batch, 2, index) = 2.0f;
			*result.tensor.ptr<float>(batch, 3, index) = 2.0f;
			if (segmentation) {
				*result.tensor.ptr<float>(batch, 6, index) = index == 1 ? -2.0f : 2.0f;
			}
		}
	}
	*result.tensor.ptr<float>(0, 4, 0) = 0.9f;
	*result.tensor.ptr<float>(0, 4, 1) = 0.8f;
	*result.tensor.ptr<float>(0, 5, 2) = 0.7f;
	*result.tensor.ptr<float>(1, 5, 1) = 0.6f;
	return result;
}

template<typename Result>
void checkItem(const Result& item, const cv::Rect& box, float score) {
	check(item.boxs == box);
	check(std::abs(item.scores - score) < 0.0001f);
}

template<typename Policy>
void testParsers() {
	using ParseImpl = dp::LetterBoxParseImpl<dp::LetterBox<>>;
	const ParseImpl letterboxes(2, dp::LetterBox<>(cv::Size(16, 16), cv::Size(16, 16)));
	const std::vector<float> thresholds{ 0.5f, 0.5f };
	dp::Yolov8DetectLetterBoxResultParser<ParseImpl, Policy> detectParser;
	auto detections = detectParser.parse({ predictions(false) }, letterboxes, 2, thresholds);
	checkAccess(detections);
	check(detections.size() == 2 && detections.batchAt(0).size() == 2);
	check(detections.batchClassAt(0, 0).size() == 2);
	checkItem(detections.batchClassAt(0, 0)[0], cv::Rect(3, 3, 2, 2), 0.9f);
	checkItem(detections.batchClassAt(0, 0)[1], cv::Rect(7, 3, 2, 2), 0.8f);
	checkItem(detections.batchClassAt(0, 1).at(0), cv::Rect(11, 3, 2, 2), 0.7f);
	check(detections.batchClassAt(1, 0).empty());
	checkItem(detections.batchClassAt(1, 1).at(0), cv::Rect(7, 3, 2, 2), 0.6f);

	const int protoDimensions[] = { 2, 1, 16, 16 };
	dp::Tensor protos{ { "protos", { 2, 1, 16, 16 } },
		cv::Mat(4, protoDimensions, CV_32F, cv::Scalar(1)) };
	dp::Yolov8SegmentLetterBoxResultParser<ParseImpl, Policy> segmentParser;
	auto segments = segmentParser.parse({ predictions(true), protos }, letterboxes, 2, thresholds);
	checkAccess(segments);
	check(segments.size() == 2 && segments.batchAt(0).size() == 2);
	check(segments.batchClassAt(0, 0).size() == 2);
	const auto& first = segments.batchClassAt(0, 0)[0];
	const auto& second = segments.batchClassAt(0, 0)[1];
	checkItem(first, cv::Rect(3, 3, 2, 2), 0.9f);
	checkItem(second, cv::Rect(7, 3, 2, 2), 0.8f);
	check(first.masks.type() == CV_8UC1 && first.masks.size() == first.boxs.size());
	check(second.masks.type() == CV_8UC1 && second.masks.size() == second.boxs.size());
	check(first.masks.at<unsigned char>(0, 0) > 200);
	check(second.masks.at<unsigned char>(0, 0) < 50);
	checkItem(segments.batchClassAt(0, 1).at(0), cv::Rect(11, 3, 2, 2), 0.7f);
	check(segments.batchClassAt(1, 0).empty());
	checkItem(segments.batchClassAt(1, 1).at(0), cv::Rect(7, 3, 2, 2), 0.6f);
}

}

int main() {
	static_assert(std::is_same_v<decltype(std::declval<dp::DetectResArray&>().batchClassAt(0, 0)),
		std::vector<dp::DetectRes>&>);
	static_assert(std::is_same_v<decltype(std::declval<const dp::SegmentResArray&>().batchClassAt(0, 0)),
		const std::vector<dp::SegmentRes>&>);
	testParsers<dp::NmsPolicy>();
	testParsers<dp::NmmPolicy>();
}
