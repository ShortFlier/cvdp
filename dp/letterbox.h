#pragma once

#include "dputility.h"

#include "dp.h"

/*
	@srcMat 输入图像
	@targetSize 目标尺寸
	@autoShape =true将 dw/dh 向下对齐到 stride 的整数倍，保证模型下采样时尺寸整除
	@scaleFill =true直接拉伸图像到目标尺寸，不保留宽高比
	@scaleUp =false只缩小不放大
	@stride 对齐值
	@color 填充颜色

	LetterBox 使用非类型模板参数绑定预处理配置，确保预处理与结果解析使用相同的参数。
*/
template<bool autoShape = false, bool scaleFill = false, bool scaleUp = false, int stride = 32>
class LetterBox {
public:
	LetterBox() = default;

	LetterBox(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color = cv::Scalar::all(0));

	void set(cv::Size srcSize, cv::Size targetSize,
		const cv::Scalar& color);

	// [ratio_x, ratio_y, pad_x, pad_y]
	const cv::Vec4d& params() const {
		return _params;
	}

	const cv::Size& targetSize() const {
		return _targetSize;
	}

	const cv::Size& srcSize() const {
		return _srcSize;
	}

	cv::Mat apply(const cv::Mat& srcMat) const;

	// 将检测框从 letterbox 后的坐标系反算回原图坐标系。
	cv::Rect enRect(const cv::Rect& rect) const;

private:
	cv::Size _srcSize;
	cv::Size _targetSize;
	cv::Scalar _color;
	cv::Vec4d _params; // [ratio_x, ratio_y, pad_x, pad_y]
};

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline LetterBox<autoShape, scaleFill, scaleUp, stride>::LetterBox(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
{
	set(srcSize, targetSize, color);
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline void LetterBox<autoShape, scaleFill, scaleUp, stride>::set(cv::Size srcSize, cv::Size targetSize,
	const cv::Scalar& color)
{
	_srcSize = srcSize;
	_targetSize = targetSize;
	_color = color;

	// 先按输入图像和目标图像的最小缩放比例计算统一缩放系数，保证完整显示。
	float r = std::min((float)_targetSize.height / (float)_srcSize.height,
		(float)_targetSize.width / (float)_srcSize.width);
	if (!scaleUp) {
		// 只缩小不放大，避免小图被强行拉大带来失真。
		r = std::min(r, 1.0f);
	}

	// 记录缩放后的宽高，后续据此计算需要补边的像素数。
	float ratio[2] = { r, r };
	int new_un_pad[2] = {
		static_cast<int>(std::round((float)_srcSize.width * r)),
		static_cast<int>(std::round((float)_srcSize.height * r))
	};

	// 目标尺寸与缩放后尺寸的差值就是需要填充的空白区域。
	auto dw = static_cast<float>(_targetSize.width - new_un_pad[0]);
	auto dh = static_cast<float>(_targetSize.height - new_un_pad[1]);

	if (autoShape) {
		// 按 stride 对齐补边，确保网络下采样时尺寸可整除。
		dw = static_cast<float>(static_cast<int>(dw) % stride);
		dh = static_cast<float>(static_cast<int>(dh) % stride);
	}
	else if (scaleFill) {
		// 直接拉伸到目标尺寸，不保留原始宽高比。
		dw = 0.0f;
		dh = 0.0f;
		new_un_pad[0] = _targetSize.width;
		new_un_pad[1] = _targetSize.height;
		ratio[0] = static_cast<float>(_targetSize.width) / (float)_srcSize.width;
		ratio[1] = static_cast<float>(_targetSize.height) / (float)_srcSize.height;
	}

	// 左右、上下各分一半，保证补边居中。
	dw /= 2.0f;
	dh /= 2.0f;

	// 保存缩放比例和左上角补边量，供结果坐标反算使用。
	int top = static_cast<int>(std::round(dh - 0.1f));
	int left = static_cast<int>(std::round(dw - 0.1f));

	_params = cv::Vec4d(ratio[0], ratio[1], left, top);
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline cv::Mat LetterBox<autoShape, scaleFill, scaleUp, stride>::apply(const cv::Mat& srcMat) const
{
	const int newW = static_cast<int>(std::round(_srcSize.width * _params[0]));
	const int newH = static_cast<int>(std::round(_srcSize.height * _params[1]));
	const int padLeft = static_cast<int>(_params[2]);
	const int padTop = static_cast<int>(_params[3]);

	cv::Mat resized;
	if (srcMat.size() != cv::Size(newW, newH)) {
		cv::resize(srcMat, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);
	}
	else {
		resized = srcMat.clone();
	}

	cv::Mat dst(_targetSize.height, _targetSize.width, srcMat.type(), _color);
	cv::Rect roi(padLeft, padTop, newW, newH);
	resized.copyTo(dst(roi));

	return dst;
}

template<bool autoShape, bool scaleFill, bool scaleUp, int stride>
inline cv::Rect LetterBox<autoShape, scaleFill, scaleUp, stride>::enRect(const cv::Rect& rect) const
{
	// 将检测框从 letterbox 后的坐标系反算回原图坐标系。
	int ltx = static_cast<int>(std::floor((rect.x - _params[2]) / _params[0]));
	int lty = static_cast<int>(std::floor((rect.y - _params[3]) / _params[1]));

	int rbx = static_cast<int>(std::ceil((rect.x + rect.width - _params[2]) / _params[0]));
	int rby = static_cast<int>(std::ceil((rect.y + rect.height - _params[3]) / _params[1]));

	int width = rbx - ltx;
	int height = rby - lty;

	// 再裁剪到原图有效范围内，避免越界。
	cv::Rect oriRect(ltx, lty, width, height);
	return rectValidate(oriRect, _srcSize);
}









/*
	LetterBox预处理器
	处理后返回指定大小，并且归一化到[0,1]的浮点图像。
		LetterBoxT，实例化的LetterBox类
		index，输入尺寸在 inputSize 中的索引。
*/
template<typename LetterBoxT>
using LetterBoxParseImpl=std::vector<LetterBoxT>;

template<typename LetterBoxT, int index>
class LetterBoxPreprocessor: public PreprocessBase<LetterBoxPreprocessor<LetterBoxT, index>, LetterBoxParseImpl<LetterBoxT>>{
private:
	float _scalefactor = 1.0f/255.0f;
	cv::Scalar _fillColor = cv::Scalar(114, 114, 114);
	bool _swapRB = true;

	LetterBoxParseImpl<LetterBoxT> _letterBoxs;
public:
	void setScaleFactor(float scalefactor) {
		_scalefactor = scalefactor;
	}

	void setFillColor(const cv::Scalar& fillColor) {
		_fillColor = fillColor;
	}

	void setSwapRB(bool swapRB) {
		_swapRB = swapRB;
	}

	std::vector<Tensor> preprocess(const std::vector<cv::Mat>& srcMats,
		 							const std::vector<TensorInfo>& inputSize,
									LetterBoxParseImpl<LetterBoxT>& letterBoxs) {

		//获取模型图片输入尺寸								
		cv::Size targetSize=getImageInputSize(inputSize, index);
		
		
		Tensor tensor;
		tensor.info=inputSize[index];
		
		//创建对应LetterBox实例
		_letterBoxs.resize(srcMats.size());

		//使用LetterBox缩放指定尺寸
		std::vector<cv::Mat> letterBoxMats;
		for(int i=0; i<srcMats.size(); i++){
			LetterBoxT& _letterBox = _letterBoxs[i];
			const cv::Mat& srcMat = srcMats[i];
			_letterBox.set(srcMat.size(), targetSize, _fillColor);
			cv::Mat mat = _letterBox.apply(srcMat);

			letterBoxMats.push_back(mat);
		}

		letterBoxs=_letterBoxs;

		//转为张量
		tensor.tensor=cv::dnn::blobFromImages(letterBoxMats, _scalefactor, cv::Size(), cv::Scalar(), _swapRB, false);
		tensor.info.shape={tensor.tensor.size[0], tensor.tensor.size[1], tensor.tensor.size[2], tensor.tensor.size[3]};			

		return std::vector<Tensor>{tensor};
	}


};
