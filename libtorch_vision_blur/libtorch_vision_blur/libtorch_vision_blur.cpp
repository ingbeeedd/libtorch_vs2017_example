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
	
	std::cout << img.size() << std::endl;
	std::cout << img.channels() << std::endl;

	auto sample = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte).clone();
	sample = sample.permute({ (2), (0), (1) }).unsqueeze(0);

	// fix seed
	torch::manual_seed(42);
	torch::cuda::manual_seed(42);
	torch::cuda::manual_seed_all(42);

	// auto sample = torch::randn({  3, 224, 224 }).unsqueeze(0);
	int64_t kernel = 5;
	
	// sample = torch::nn::functional::pad(sample, torch::nn::functional::PadFuncOptions({ 1, 1, 1, 1 }).mode(torch::kReflect));

	const auto padding_left_right = (kernel - 1) / 2;
	const auto padding_top_bottom = (kernel - 1) / 2;

	std::cout << "sample: " << sample.sizes() << std::endl;

	/*auto padded_input = torch::replication_pad2d(sample,
		{ padding_left_right, padding_left_right,
		 padding_top_bottom, padding_top_bottom});*/

	auto padded_input = torch::nn::functional::pad(sample, torch::nn::functional::PadFuncOptions(
															{ padding_left_right, padding_left_right,
															 padding_top_bottom, padding_top_bottom }).mode(torch::kConstant));

	std::cout << "padded_input: " << padded_input.sizes() << " " << padded_input.dtype() << std::endl;

#if 1
	auto x = padded_input.unfold(2, kernel, 1).unfold(3, kernel, 1);

	std::cout << "x: " << x.sizes() << " " << x.min() << " " << x.max() << std::endl;

	x = std::get<0>(x.contiguous().view({ x.size(0), x.size(1), x.size(2), x.size(3), kernel * kernel}).median(-1));

	std::cout << "x: " << x.sizes() << " " << x.min() << " " << x.max() << std::endl;

#endif
	auto torch2cv = ToCvImage(transpose(x.squeeze(0), { (1),(2),(0) }));
	
	std::cout << "torch2cv: " << torch2cv.size() << " " << std::endl;

	cv::imshow("blur", torch2cv);
	cv::imshow("original", ToCvImage(transpose(padded_input.squeeze(0), { (1),(2),(0) })));
	cv::waitKey(0);


    return 0;
}