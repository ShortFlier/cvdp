#pragma once

#include <opencv2/opencv.hpp>

#include "log.h"


#include <chrono>

#define ELAPSED(str, exe)                                                     \
    do {                                                                      \
        auto __start = std::chrono::high_resolution_clock::now();             \
        exe;                                                                   \
        auto __end = std::chrono::high_resolution_clock::now();               \
        auto __duration = std::chrono::duration_cast<std::chrono::microseconds>(__end - __start).count(); \
        log_info("{0}耗时：{1}ms", str, __duration / 1000.0); \
    } while(0)


/*
	检测结果结构体，包含该类别的所有检测框和分数
	boxs：检测框坐标数组
	scores：检测分数数组
*/
struct DetectRes
{
	std::vector<cv::Rect> boxs;
	std::vector<float> scores;
};
// 检测结果数组，外层vector表示图像批次，内层vector表示类别检测结果
typedef std::vector<std::vector<DetectRes>> DetectResArray;

/*
	分割结果结构体，包含该类别的所有检测框、分数和掩膜
	boxs：检测框坐标数组
	scores：检测分数数组
	masks：分割掩膜数组，大小为对应box区域的大小
*/
struct SegmentRes {
	std::vector<cv::Rect> boxs;
	std::vector<float> scores;
	std::vector<cv::Mat> masks;//Mask为box区域的掩膜

};
// 分割结果数组，外层vector表示图像批次，内层vector表示类别分割结果
typedef std::vector<std::vector<SegmentRes>> SegmentResArray;


//张量名称、形状信息
struct TensorInfo{
	std::string name;
    std::vector<int64_t> shape;
};

//张量信息和实际数据
struct Tensor{
	TensorInfo info;
	cv::Mat tensor;
};

/*
	深度学习模型加载类
		load,函数loadImpl，加载模型
		get,函数getImpl，返回可用模型
		getSize,函数getSizeImpl， 返回模型输入输出尺寸
*/

template<typename _ModelLoader, typename _Model>
class ModelLoaderBase{
public:
	void load(const char* path, const char* cfg = nullptr){
		static_cast<_ModelLoader*>(this)->loadImpl(path, cfg);
	}

	_Model& get(){
		return static_cast<_ModelLoader*>(this)->getImpl();
	}

	//返回模型声明的输入、输出张量大小，带-1表示动态维度，需要在图片预处理或者推理后获取
	void getSize(std::vector<TensorInfo>& inputTensorInfos, std::vector<TensorInfo>& outputTensorInfos){
		static_cast<_ModelLoader*>(this)->getSizeImpl(inputTensorInfos, outputTensorInfos);
	}
};


/*
	@图像预处理器，对原始输入图像进行预处理，返回模型所需的输入张量，返回结果解析辅助接口
	@_Preprocess，图像预处理类
	@_ParseImpl, 结果解析辅助接口，在结果解析恢复box框大小等时候使用
	@operator(),函数preprocess，图像预处理，预备输入数据，转为张量输入集
*/
template<typename _Preprocess, typename _ParseImpl>
class PreprocessBase{
public:

	using ParseImplType=_ParseImpl;

	/*
		operator()，函数preprocess，图像预处理，预备输入数据，转为张量输入集
		@src 输入图像数组
		@inputTensorInfos 模型输入张量信息
		@parseImpl 输出，结果解析辅助接口
	*/
	std::vector<Tensor> operator()( const std::vector<cv::Mat>& src,
									const std::vector<TensorInfo>& inputTensorInfos,
									ParseImplType& parseImpl ){
		return static_cast<_Preprocess*>(this)->preprocess(src, inputTensorInfos, parseImpl);
	}

};


/*
	模型运行器
		operator()，函数run，使用模型进行推理，返回结果张量
*/
template<typename _Runner, typename _Model>
class RunnerBase{
public:
	//使用模型进行推理，返回结果张量
	std::vector<Tensor> operator()(_Model& model, const std::vector<Tensor>& inputTensors){
		return static_cast<_Runner*>(this)->run(model, inputTensors);
	}

};


/*
	结果解析器
		operator()，函数parse，解析模型输出的结果张量，返回结果数组

*/
template<typename _Parser, typename _ParseImpl, typename _Result>
class ParserBase{
public:
	/*
		operator()，函数parse，解析模型输出的结果张量，返回结果数组
		@outputs 模型输出的结果张量
		@parseImpl 结果解析辅助接口，在结果解析恢复box框大小等时候使用
		@classNum 类别数
		@socreThreshs 分数阈值
		@nmsThreshs NMS阈值
	*/
	typename _Result operator()(const std::vector<Tensor>& outputs,
								const _ParseImpl& parseImpl,
								int classNum,
								const std::vector<float>& socreThreshs, 
								const std::vector<float>& nmsThreshs){
		return static_cast<_Parser*>(this)->parse(outputs, parseImpl, classNum, socreThreshs, nmsThreshs);
	}

};

/*
	深度学习检测模型功能类

	@_ModelLoader 模型加载器
		参考 ModelLoaderBase 自定义，必须可以调用load，get，getSize函数



	@_Preprocess图像预处理器
		必须是可以调用的函数对象，参考 PreprocessBase


	@_Runner运行器，返回结果张量
		必须是可以调用的函数对象，参考 RunnerBase


	@_Parser解析器，，返回结果数组
		必须是可以调用的函数对象，参考 ParserBase

	@_Result结果数组类型，DetectResArray或SegmentResArray
*/
template<typename _ModelLoader,
	typename _Preprocess,
	typename _Runner,
	typename _Parser,
	typename _Result>
class _DPBase {
	public:
		_ModelLoader _modelLoader;
		_Preprocess _preprocess;
		_Runner _runner;
		_Parser _parser;

		//输入张量大小
		std::vector<TensorInfo> _inputTensorInfos;
		//输出张量大小
		std::vector<TensorInfo> _outputTensorInfos;

		//类别数
		int _classNum;
		//分数阈值
		std::vector<float> _threshs;
		//NMS阈值
		std::vector<float> _nmsThreshs;

		

	public:

		//传入的分数阈值或NMS阈值为空，分别设置为默认的0.5、0.4
		//传入的分数阈值或NMS阈值数量不足classNum，使用默认值补全
		_DPBase(int classNum, const std::vector<float>& threshs = std::vector<float>(),
			const std::vector<float>& nmsThreshs = std::vector<float>()){
				setClassNum(classNum);
				setScoreThreshs(threshs);
				setNmsThreshs(nmsThreshs);

			log_info("DP推理参数classNum: {0}, threshs: {1}, nmsThreshs: {2}",
				 _classNum, fmt::join(_threshs, ", "), fmt::join(_nmsThreshs, ", "));
		}

		void setClassNum(int classNum) {
			_classNum = classNum;
		}

		void setScoreThreshs(const std::vector<float>& threshs) {
			_threshs.clear();
			_threshs = threshs;
			int needThresh = _classNum - static_cast<int>(_threshs.size());
			for (int i = 0; i < needThresh; ++i) {
				_threshs.push_back(0.5);
			}
		}

		void setNmsThreshs(const std::vector<float>& nmsThreshs) {
			_nmsThreshs.clear();
			_nmsThreshs = nmsThreshs;
			int needNms = _classNum - static_cast<int>(_nmsThreshs.size());
			for (int i = 0; i < needNms; ++i) {
				_nmsThreshs.push_back(0.4f);
			}
		}

		//加载模型
		void loadModel(const char* path, const char* cfg=nullptr) {
			log_info("DP加载模型: {0}", path);

			_modelLoader.load(path, cfg);

			//获取模型输入、输出信息
			_modelLoader.getSize(_inputTensorInfos, _outputTensorInfos);

			
			log_info("模型输入数：{0}: ", _inputTensorInfos.size());
			for (size_t i = 0; i < _inputTensorInfos.size(); ++i) {
				log_info("输入{0}，{1}: [{2}]", i, _inputTensorInfos[i].name, fmt::join(_inputTensorInfos[i].shape, ","));
			}
			
			log_info("模型输出数：{0}: ", _outputTensorInfos.size());
			for (size_t i = 0; i < _outputTensorInfos.size(); ++i) {
				log_info("输出{0}，{1}: [{2}]", i, _outputTensorInfos[i].name, fmt::join(_outputTensorInfos[i].shape, ","));
			}
		}

		//推理，获取结果
		_Result run(const std::vector<cv::Mat>& srcMats) {
			_Result res;

			ELAPSED("总推理",

				//图像预处理
				log_info("执行图像预处理");
				typename _Preprocess::ParseImplType parseImpl;
				std::vector<Tensor> inputDatas = _preprocess(srcMats, _inputTensorInfos, parseImpl);

				log_debug("图像预处理完成，输入张量数量: {0}", inputDatas.size());
				for (size_t i = 0; i < inputDatas.size(); ++i) {
					log_debug("输入张量{0}, {1}: [{2}]", i, inputDatas[i].info.name, fmt::join(inputDatas[i].info.shape, ","));
				}

				//运行
				log_info("执行模型推理");
				std::vector<Tensor> resTensor;
				ELAPSED("模型推理", resTensor = _runner(_modelLoader.get(), inputDatas));

				log_debug("模型推理完成，输出张量数量: {0}", resTensor.size());
				for (size_t i = 0; i < resTensor.size(); ++i) {
					log_debug("输出张量{0}, {1}: [{2}]", i, resTensor[i].info.name, fmt::join(resTensor[i].info.shape, ","));
				}

				//解析结果
				log_info("解析结果");
				ELAPSED("结果解析", res = _parser(resTensor, parseImpl, _classNum, _threshs, _nmsThreshs));
			);

			return res;
		}
};

template<typename _ModelLoader,
	typename _Preprocess,
	typename _Runner,
	typename _Parser>
using DPDetector = _DPBase<_ModelLoader, _Preprocess, _Runner, _Parser, DetectResArray>;

template<typename _ModelLoader,
	typename _Preprocess,
	typename _Runner,
	typename _Parser>
using DPSegmentor = _DPBase<_ModelLoader, _Preprocess, _Runner, _Parser, SegmentResArray>;