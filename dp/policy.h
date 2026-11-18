#pragma once

#include "dp.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define DP_DEFAULT_IOU_THRESHOLD 0.4f
#define DP_DEFAULT_MASK_THRESHOLD 0.6f

NAMESPACE_DP_BEGIN

struct BoxCandidate {
	cv::Rect box;
	float score;
	int predictionIndex;
};

struct BoxCandidateGroup {
	cv::Rect box;
	float score;
	std::vector<BoxCandidate> members;
};


//非极大值抑制
class NmsPolicy {
public:
	std::vector<BoxCandidateGroup> apply(const std::vector<BoxCandidate>& candidates,
										float scoreThreshold,
										float iouThreshold) const {
		std::vector<cv::Rect> boxes;
		std::vector<float> scores;
		boxes.reserve(candidates.size());
		scores.reserve(candidates.size());

		for (const auto& candidate : candidates) {
			boxes.push_back(candidate.box);
			scores.push_back(candidate.score);
		}

		std::vector<int> indices;
		cv::dnn::NMSBoxes(boxes, scores, scoreThreshold, iouThreshold, indices);

		std::vector<BoxCandidateGroup> groups;
		groups.reserve(indices.size());
		for (int index : indices) {
			const auto& candidate = candidates[index];
			groups.push_back({ candidate.box, candidate.score, { candidate } });
		}
		return groups;
	}
};


//非极大值合并
class NmmPolicy {
public:
	std::vector<BoxCandidateGroup> apply(const std::vector<BoxCandidate>& candidates,
										float scoreThreshold,
										float iouThreshold) const {
		std::vector<int> order;
		order.reserve(candidates.size());
		for (size_t i = 0; i < candidates.size(); ++i) {
			if (candidates[i].score >= scoreThreshold) {
				order.push_back(static_cast<int>(i));
			}
		}

		std::stable_sort(order.begin(), order.end(), [&candidates](int lhs, int rhs) {
			if (candidates[lhs].score != candidates[rhs].score) {
				return candidates[lhs].score > candidates[rhs].score;
			}
			return candidates[lhs].predictionIndex < candidates[rhs].predictionIndex;
		});

		std::vector<bool> grouped(candidates.size(), false);
		std::vector<BoxCandidateGroup> groups;
		for (int seedIndex : order) {
			if (grouped[seedIndex]) {
				continue;
			}

			std::vector<BoxCandidate> members;
			members.push_back(candidates[seedIndex]);
			grouped[seedIndex] = true;

			for (int candidateIndex : order) {
				if (grouped[candidateIndex]) {
					continue;
				}
				if (intersectionOverUnion(candidates[seedIndex].box, candidates[candidateIndex].box) > iouThreshold) {
					members.push_back(candidates[candidateIndex]);
					grouped[candidateIndex] = true;
				}
			}

			groups.push_back({ weightedBox(members), candidates[seedIndex].score, std::move(members) });
		}

		return groups;
	}

private:
	static double intersectionOverUnion(const cv::Rect& lhs, const cv::Rect& rhs) {
		const cv::Rect intersection = lhs & rhs;
		const double intersectionArea = static_cast<double>(intersection.area());
		const double unionArea = static_cast<double>(lhs.area()) + rhs.area() - intersectionArea;
		return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
	}

	static cv::Rect weightedBox(const std::vector<BoxCandidate>& members) {
		double totalWeight = 0.0;
		double x = 0.0;
		double y = 0.0;
		double width = 0.0;
		double height = 0.0;

		for (const auto& member : members) {
			const double weight = std::max(static_cast<double>(member.score), 0.0);
			totalWeight += weight;
			x += member.box.x * weight;
			y += member.box.y * weight;
			width += member.box.width * weight;
			height += member.box.height * weight;
		}

		if (totalWeight == 0.0) {
			totalWeight = static_cast<double>(members.size());
			for (const auto& member : members) {
				x += member.box.x;
				y += member.box.y;
				width += member.box.width;
				height += member.box.height;
			}
		}

		return cv::Rect(
			static_cast<int>(std::lround(x / totalWeight)),
			static_cast<int>(std::lround(y / totalWeight)),
			std::max(1, static_cast<int>(std::lround(width / totalWeight))),
			std::max(1, static_cast<int>(std::lround(height / totalWeight))));
	}
};

struct BoxMat {
	cv::Rect box;
	cv::Mat mat;
};

//掩膜合并
class OrMatPolicy {
public:
	bool apply(const std::vector<BoxMat>& inputs, BoxMat& result) const {
		if (inputs.empty()) {
			result = {};
			return false;
		}

		cv::Rect unionBox;
		for (const auto& input : inputs) {
			if (input.mat.empty()
				|| input.mat.type() != CV_8UC1
				|| input.box.empty()
				|| input.mat.size() != input.box.size()) {
				result = {};
				return false;
			}
			unionBox = unionBox.empty() ? input.box : unionBox | input.box;
		}

		cv::Mat merged(unionBox.size(), CV_8UC1, cv::Scalar(0));
		for (const auto& input : inputs) {
			const cv::Rect destination(
				input.box.x - unionBox.x,
				input.box.y - unionBox.y,
				input.box.width,
				input.box.height);
			cv::Mat region = merged(destination);
			cv::bitwise_or(region, input.mat, region);
		}
		result = { unionBox, std::move(merged) };
		return true;
	}
};

NAMESPACE_DP_END
