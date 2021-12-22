// 1. libtorch + c++17 Issue
// https://discuss.pytorch.org/t/libtorch-c-17-visual-studio/73793
// 2. CUDA Not Found Issue
// https://github.com/pytorch/pytorch/issues/37124
// 3. Validation Accuracy Problem
// 3-1. MNIST, CIFAR10 Test하기 (코드 문제)
// 3-2. class imbalance 문제

#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>

#include "scheduler.h"
#include "mnist.h"
#include "transform.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;
using transform::RandomVerticalFlip;

auto transpose(at::Tensor tensor, c10::IntArrayRef dims = { 0, 3, 1, 2 }) {
	tensor = tensor.permute(dims);
	return tensor;
}

cv::Mat ToCvImage(at::Tensor tensor, int n_channel = 1) {
	/*std::cout << "Transpose : " << tensor.sizes() << " " << tensor.dtype() << std::endl;*/

	int width = tensor.sizes()[0];
	int height = tensor.sizes()[1];
	//tensor = tensor.squeeze().detach().mul(255).clamp(0, 255);
	//tensor = tensor.to(torch::kUInt8);
	uchar type_color;
	if (n_channel == 3)
		type_color = CV_8UC3;
	else
		type_color = CV_8UC1;
	try {
		
		cv::Mat output_mat(cv::Size(width, height), type_color, tensor.data_ptr()); // initialize a Mat object <uchar>
		return output_mat.clone();
	}
	catch (const c10::Error& e) {
		std::cout << "Error OpenCV conversion : """ << e.msg() << "" << std::endl;
		return cv::Mat(height, width, CV_8UC1);
	}
}

std::string type2str(int type) {
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

int main()
{
	// check cuda available
	std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
	 
	auto imagepath = "./datasets/classification/train/dotparticle/DP0001.png";
	auto img = cv::imread(imagepath, cv::IMREAD_COLOR);
	cv::resize(img, img, cv::Size(224, 224));
	
	double minval, maxval;
	std::cout << "img size: " << img.size() << std::endl;
	std::cout << "img channel: " << img.channels() << std::endl;
	std::cout << "img type: " << img.type() << std::endl;
	std::cout << "img type(str): " << type2str(img.type()) << std::endl;
	
	cv::minMaxLoc(img, &minval, &maxval);

	std::cout << "img min: " << minval << std::endl;
	std::cout << "img max: " << maxval << std::endl;

	cv::medianBlur(img, img, 5);

	cv::imshow("blur", img);
	cv::waitKey(0);

	auto sample = torch::zeros({ img.rows, img.cols, img.channels() }, torch::kUInt8);

	std::memcpy(sample.data_ptr(), img.data, sample.numel());
	
	std::cout << "sample size: " << sample.sizes() << std::endl;
	std::cout << "sample dtype: " << sample.dtype() << std::endl; 
	std::cout << "sample size(dtype): " << sizeof(sample.dtype()) << std::endl;
	std::cout << "sample min: " << sample.min() << std::endl;
	std::cout << "sample max: " << sample.max() << std::endl;

	sample = sample.permute({ (2), (0), (1) });

	// fix seed
	torch::manual_seed(42);
	torch::cuda::manual_seed(42);
	torch::cuda::manual_seed_all(42);

	// auto sample = torch::randn({ 3, 224, 224 }).unsqueeze(0);
	int64_t kernel = 5;
	
	cv::convertScaleAbs()
	// memcpy
	cv::Mat sample_mat(sample.size(1), sample.size(2), CV_8UC3);
	std::memcpy(sample_mat.data, sample.data_ptr(), sample.numel());
	
	cv::minMaxLoc(sample_mat, &minval, &maxval);

	std::cout << "sample_mat size: " << sample_mat.size() << " " << std::endl;
	std::cout << "sample_mat dim: " << sample_mat.channels() << " " << std::endl;
	std::cout << "sample_mat min: " << minval << std::endl;
	std::cout << "sample_mat max: " << maxval << std::endl;

	
	cv::imshow("origin", sample_mat);
	cv::waitKey(0);


	// sample = torch::nn::functional::pad(sample, torch::nn::functional::PadFuncOptions({ 1, 1, 1, 1 }).mode(torch::kReflect));

	// const auto padding_left_right = (kernel - 1) / 2;
	// const auto padding_top_bottom = (kernel - 1) / 2;

	/*auto padded_input = torch::replication_pad2d(sample,
		{ padding_left_right, padding_left_right,
		 padding_top_bottom, padding_top_bottom});*/

	//auto padded_input = torch::nn::functional::pad(sample, torch::nn::functional::PadFuncOptions(
	//														{ padding_left_right, padding_left_right,
	//														 padding_top_bottom, padding_top_bottom }).mode(torch::kConstant));

	//auto x = padded_input.unfold(2, kernel, 1).unfold(3, kernel, 1);

	//std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	//x = std::get<0>(x.contiguous().view({ x.size(0), x.size(1), x.size(2), x.size(3), kernel * kernel}).median(-1));
	//// x = std::get<0>(x.reshape({ x.size(0), x.size(1), x.size(2), x.size(3), kernel * kernel }).median(-1));
	//std::cout << "median blur 수행 시간 : " << static_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start).count() << " seconds" << std::endl;

	/*auto torch2cv = ToCvImage(transpose(x.squeeze(0), { (1),(2),(0) }));
	
	std::cout << "torch2cv: " << torch2cv.size() << " " << std::endl;

	cv::imshow("blur", torch2cv);
	cv::imshow("original", ToCvImage(transpose(padded_input.squeeze(0), { (1),(2),(0) })));
	cv::waitKey(0);*/

    return 0;
}