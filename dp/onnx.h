#pragma once

#include <onnxruntime_cxx_api.h>

#include "dputility.h"

#include "windows.h"

#include "log.h"


/*
* onnxruntime的模型加载
* _usingGPU: 是否使用GPU推理
* _intraConcurrency: CPU推理时，intra并发数，设置0时，设置为当前CPU线程数一半
* _interConcurrency: CPU推理时，inter并发数，0时不设置inter并发
*/
class OnnxLoader {

private:
	// Env must outlive Session, so declare Env before Session.
	Ort::Env env;
	Ort::Session session;

	bool _usingGPU;
	unsigned short _intraConcurrency;
	unsigned short _interConcurrency;

public:
	OnnxLoader(bool usingGPU=true):env(nullptr),session(nullptr),_usingGPU(usingGPU),_intraConcurrency(0),_interConcurrency(0){};

	void setUsingGPU(bool usingGPU) {
		_usingGPU = usingGPU;
	}

	void setIntraConcurrency(unsigned short intraConcurrency) {
		_intraConcurrency = intraConcurrency;
	}

	void setInterConcurrency(unsigned short interConcurrency) {
		_interConcurrency = interConcurrency;
	}
	
	//返回会话
	Ort::Session& get(){
		return session;
	}


	//加载模型
	void load(const char* path, const char* cfg = nullptr);

	//返回输入、输出张量大小
	void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes);
};




//onnxruntime的单输入运行推理
class SingleInputOnnxRunner {
public:
	SingleInputOnnxRunner() {}

	std::vector<cv::Mat> operator()(Ort::Session& session, cv::Mat blob);
};
