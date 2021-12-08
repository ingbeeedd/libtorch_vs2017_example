#pragma once

#include <torch/torch.h>
#include <random>
#include <vector>

namespace transform 
{
	class RandomHorizontalFlip : public torch::data::transforms::TensorTransform<torch::Tensor> 
	{
	public:

		explicit RandomHorizontalFlip(double p = 0.5);

		torch::Tensor operator()(torch::Tensor input) override;

	private:
		double p_;
	};

	class RandomVerticalFlip : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:

		explicit RandomVerticalFlip(double p = 0.5);

		torch::Tensor operator()(torch::Tensor input) override;

	private:
		double p_;
	};

	class RandomCrop : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:

		explicit RandomCrop(const std::vector<int64_t>& size);
		torch::Tensor operator()(torch::Tensor input) override;

	private:
		std::vector<int64_t> size_;

	};

	class ConstantPad : public torch::data::transforms::TensorTransform<torch::Tensor>
	{
	public:
		explicit ConstantPad(const std::vector<int64_t>& padding, torch::Scalar value = 0);
		explicit ConstantPad(int64_t padding, torch::Scalar value = 0);
		torch::Tensor operator()(torch::Tensor input) override;

	private:
		std::vector<int64_t> padding_;
		torch::Scalar value_;
	};
}