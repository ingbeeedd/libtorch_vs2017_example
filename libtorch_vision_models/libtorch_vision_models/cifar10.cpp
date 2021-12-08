#include "cifar10.h"

namespace {

	constexpr uint32_t kTrainSize = 50000;
	constexpr uint32_t kTestSize = 10000;
	constexpr uint32_t kSizePerBatch = 10000;
	constexpr uint32_t kImageRows = 32;
	constexpr uint32_t kImageColumns = 32;
	constexpr uint32_t kBytesPerRow = 3073; // include target
	constexpr uint32_t kBytesPerChannelPerRow = 1024;
	constexpr uint32_t kBytesPerBatchFile = kBytesPerRow * kSizePerBatch;

	const std::vector<std::string> kTrainDataBatchFiles = {
		"data_batch_1.bin",
		"data_batch_2.bin",
		"data_batch_3.bin",
		"data_batch_4.bin",
		"data_batch_5.bin",
	};

	const std::vector<std::string> kTestDataBatchFiles = {
		"test_batch.bin" };


	std::string join_paths(std::string head, const std::string& tail) {
		if (head.back() != '/') {
			head.push_back('/');

		}

		head += tail;

		return head;
	}

	std::pair<torch::Tensor, torch::Tensor> read_data(const std::string& root, bool train)
	{
		const auto& files = train ? kTrainDataBatchFiles : kTestDataBatchFiles;
		const auto num_samples = train ? kTrainSize : kTestSize;

		std::vector<char> data_buffer;
		data_buffer.reserve(files.size() * kBytesPerBatchFile);

		for (const auto& file : files) {
			const auto path = join_paths(root, file);
			std::ifstream data(path, std::ios::binary);

			// cond, VA_ARGS
			TORCH_CHECK(data, "Error opening data files at", path);

			// insert
			data_buffer.insert(data_buffer.end(), std::istreambuf_iterator<char>(data), {});
		}

		TORCH_CHECK(data_buffer.size() == files.size() * kBytesPerBatchFile, "Unexpected file sizes");

		auto targets = torch::empty(num_samples, torch::kByte);
		auto images = torch::empty({ num_samples, 3, kImageRows, kImageColumns }, torch::kByte);

		for (uint32_t i = 0; i != num_samples; ++i) {

			uint32_t start_index = i * kBytesPerRow;
			targets[i] = data_buffer[start_index];

			uint32_t image_start = start_index + 1;
			uint32_t image_end = image_start + 3 * kBytesPerChannelPerRow;
			std::copy(data_buffer.begin() + image_start, data_buffer.begin() + image_end,
				// 임의의 포인터 타입 변환
				reinterpret_cast<char*>(images[i].data_ptr()));
		}

		return { images.to(torch::kFloat32).div_(255), targets.to(torch::kInt64) };
	}

}

CIFAR10::CIFAR10(const std::string& root, Mode mode) : mode_(mode)
{
	auto data = read_data(root, mode == Mode::kTrain);

	images_ = std::move(data.first);
	targets_ = std::move(data.second);
}

torch::data::Example<> CIFAR10::get(size_t index)
{
	return { images_[index], targets_[index] };
}

torch::optional<size_t> CIFAR10::size() const
{
	return images_.size(0);
}

bool CIFAR10::is_train() const noexcept
{
	return mode_ == Mode::kTrain;
}

const torch::Tensor& CIFAR10::images() const
{
	return images_;
}

const torch::Tensor& CIFAR10::targets() const
{
	return targets_;
}
