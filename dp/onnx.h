#pragma once

#include <onnxruntime_cxx_api.h>

#include "dputility.h"

#include "windows.h"

#include "log.h"


/*
* onnxruntime的模型加载
* _deviceType: 使用推理设备，没有可以设备时，使用CPU推理
* _intraConcurrency: CPU推理时，intra并发数，设置0时，设置为当前CPU线程数一半
* _interConcurrency: CPU推理时，inter并发数，0时不设置inter并发
*/
class OnnxLoader{

	
public:
	//使用推理设备
	enum DeviceType{CPU, NV_CUDA, OpenVINO_CPU};
	
	OnnxLoader(DeviceType deviceType=NV_CUDA):env(nullptr),session(nullptr){
		setDeviceType(deviceType);
		setCPUParams(0, 0);
		setOpenVINOCPUParams(0);
	};
	

	//设置使用推理设备
	void setDeviceType(DeviceType deviceType){
		_deviceType=deviceType;
	}
	
	/*
	* 设置使用cpu推理时的参数
	*@intraConcurrency 设置CPU推理时，intra并发数，设置0时，设置为当前CPU线程数一半
	*@interConcurrency 设置CPU推理时，inter并发数，0时不设置inter并发
	*/
	void setCPUParams(unsigned short intraConcurrency, unsigned short interConcurrency=0) {
		_intraConcurrency = intraConcurrency;
		_interConcurrency = interConcurrency;
	}

	/*
	* 设置OpenVINO推理时的参数
	*@threads，设置OpenVINO推理时的线程数，设置为0时，使用当前CPU线程数一半
	*@num_streams，设置OpenVINO推理时的流数，设置为0时，使用默认值
	*/
	void setOpenVINOCPUParams(unsigned short threads, unsigned short num_streams=0) {
		_openvinoThreads = threads;
		_openvinoNumStreams = num_streams;
	}


	/*
	* 设置CUDA推理参数
	*@deviceId 设置CUDA设备ID
	*/
	void setCUDAParams(unsigned short deviceId) {
		_cudaDeviceId = deviceId;
	}


	//返回会话
	Ort::Session& get(){
		return session;
	}

	//加载模型
	void load(const char* path, const char* cfg = nullptr);

	//返回输入、输出张量大小
	void getSize(cv::Vec4i& inputSize, std::vector<std::vector<int>>& outputSizes);

private:
	// Env must outlive Session, so declare Env before Session.
	Ort::Env env;
	Ort::Session session;

	DeviceType _deviceType;

	//没有可以设备时，使用CPU推理
	void setSessionOptions(Ort::SessionOptions& session_options);
	
	//cpu推理时使用的参数
	unsigned short _intraConcurrency=0;
	unsigned short _interConcurrency=0;

	//OpenVINO推理时使用的参数
	unsigned short _openvinoThreads=0;
	unsigned short _openvinoNumStreams=0;

	//CUDA推理时使用的参数
	unsigned short _cudaDeviceId=0;
};




//onnxruntime的单输入运行推理
class SingleInputOnnxRunner {
public:
	SingleInputOnnxRunner() {}

	std::vector<cv::Mat> operator()(Ort::Session& session, cv::Mat blob);
};
