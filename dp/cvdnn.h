#pragma once

#include "dputility.h"
#include <opencv2/dnn.hpp>

#include "dp.h"

/*	opencv::dnn模块读取onnx模型的模型读取器
	使用CPU推理
	concurrency并发数，默认0时使用当前可用线程数的一半
*/
typedef cv::dnn::Net DnnNet;


class CVDnnLoaderCPU: public ModelLoaderBase<CVDnnLoaderCPU, DnnNet> {
public:
	CVDnnLoaderCPU(){
		_inputSize=nullptr;
		_concurrency=0;
	}
	~CVDnnLoaderCPU(){
		delete _inputSize;
	}

	using _Model=DnnNet;

	//加载模型
	void loadImpl(const char* path, const char* cfg = nullptr){
		net = cv::dnn::readNetFromONNX(path);

		if(_concurrency<=0){
			_concurrency=getCPUConcurrency();
		}

		// 设置计算后端和线程数
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
		cv::setNumThreads(_concurrency);

		log_info("opencv::dnn::net推理并发数: {0}", _concurrency);
	}

	//返回模型
	_Model& getImpl(){
		return net;
	}

	void setInputSize(int batch, int channel, int height, int width){
		delete _inputSize;
		_inputSize=new cv::Vec4i(batch, channel, height, width);
	}

	//返回输入、输出张量大小
	void getSizeImpl(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes){
		//输入大小获取
		if(_inputSize==nullptr){
			const char* err="opencv::dnn::net无法直接获取输出大小，请先调用setInputSize设置输入大小!";
			log_error(err);
			throw std::runtime_error(err);
		}
		inputSize=*_inputSize;

		//输出大小获取，执行一次前向传播来获取输出大小
		int type=inputSize[1]==3?CV_8UC3:CV_8UC1;
		cv::Mat inputBlob=cv::Mat::zeros(inputSize[2], inputSize[3], type);

		net.setInput(cv::dnn::blobFromImage(inputBlob));
		std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
		std::vector<cv::Mat> outs;
		net.forward(outs, outNames);
		outputSizes.clear();
		for (const auto& out : outs) {
			std::vector<int> shape(out.size.p, out.size.p + out.dims);
			outputSizes.push_back(shape);
		}

	}

private:
	_Model net;

	cv::Vec4i* _inputSize;

	unsigned int _concurrency;	
};



class CVDNNRunner: public RunnerBase<CVDNNRunner, DnnNet> {
public:
	CVDNNRunner() {}

	std::vector<cv::Mat> run(DnnNet& net,  cv::Mat& blob){
	// 设置输入数据
	net.setInput(blob);

	//获取输出层
	std::vector<std::string> outputLayerNames = net.getUnconnectedOutLayersNames();

	// 执行推理
	std::vector<cv::Mat> outputs;
	net.forward(outputs, outputLayerNames);

	return outputs;
    }
};