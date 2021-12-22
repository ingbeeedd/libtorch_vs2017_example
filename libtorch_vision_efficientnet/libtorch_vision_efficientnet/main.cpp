// dataset 선언
// model 선언 (모델 다양하게 선택할 수 있도록)
// optimzier, scheduler, loss 선언

#include <iostream>
#include <torch/torch.h>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>
#include "cifar10.h"
#include "EfficientNet.h"

template <typename DataLoader>
void test(std::unique_ptr<vision::models::EfficientNetB0>& net,
	DataLoader& test_loader,
	torch::Device device)
{
	net->to(device);
	net->eval();

	std::cout << "Testing ... " << std::endl;

	size_t num_samples = 0;
	size_t num_correct = 0;
	float running_loss = 0.0;

	for (auto& batch : *test_loader) {
		auto data = batch.data.to(device);
		auto target = batch.target.to(device);

		num_samples += data.size(0);

		torch::Tensor output = net->forward(data);

		torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

		// loss NAN
		AT_ASSERT(!std::isnan(loss.item<float>()));

		running_loss += loss.item<float>() * data.size(0);
		auto prediction = output.argmax(1);

		num_correct += prediction.eq(target).sum().item<int64_t>();
	}

	auto sample_mean_loss = running_loss / num_samples;
	auto accuracy = static_cast<double>(num_correct) / num_samples;

	std::cout << " Testset - Loss: " << sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
}

template <typename DataLoader>
void train(std::unique_ptr<vision::models::EfficientNetB0>& net,
			int64_t num_epochs,
			DataLoader& train_loader,
			torch::optim::Optimizer &optimizer,
			torch::Device device)
{
	net->to(device);
	net->train();

	std::cout << "Training ... " << std::endl;

	for (size_t epoch = 0; epoch <= num_epochs; ++epoch)
	{
		size_t num_samples = 0;
		size_t num_correct = 0;
		float running_loss = 0.0;

		for (auto& batch : *train_loader) {
			auto data = batch.data.to(device);
			auto target = batch.target.to(device);

			num_samples += data.size(0);

			optimizer.zero_grad();

			torch::Tensor output = net->forward(data);

			torch::Tensor loss = torch::nn::functional::cross_entropy(output, target);

			// loss NAN
			AT_ASSERT(!std::isnan(loss.item<float>()));

			running_loss += loss.item<float>() * data.size(0);
			auto prediction = output.argmax(1);

			num_correct += prediction.eq(target).sum().item<int64_t>();

			loss.backward();

			optimizer.step();
		}

		auto sample_mean_loss = running_loss / num_samples;
		auto accuracy = static_cast<double>(num_correct) / num_samples;

		std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
	}
}

int main(void)
{
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

	int num_classes = 10;
	// auto net = vision::models::ResNet18(num_classes);

	vision::models::ResNet18 resnet18 = vision::models::ResNet18(num_classes);
	vision::models::ResNet50 resnet50 = vision::models::ResNet50(num_classes);
	vision::models::ResNet101 resnet101 = vision::models::ResNet101(num_classes);
	vision::models::WideResNet50_2 wresnet50_2 = vision::models::WideResNet50_2(num_classes);
	vision::models::ResNext101_32x8d resnext101_32d8 = vision::models::ResNext101_32x8d(num_classes);
	std::unique_ptr<vision::models::EfficientNetB0> net = std::make_unique<vision::models::EfficientNetB0>(num_classes);

	int64_t kTrainBatchSize = 128;
	int64_t kTestBatchSize(kTrainBatchSize);

	const std::string CIFAR10_DATASET_PATH = "./cifar-10-batches-bin/";
	const std::vector<double> norm_mean = { 0.4914, 0.4822, 0.4465 };
	const std::vector<double> norm_std = { 0.247, 0.243, 0.261 };

	auto train_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTrain)
		.map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
		.map(torch::data::transforms::Stack<>());

	const size_t train_dataset_size = train_dataset.size().value();

	auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset),
																							kTrainBatchSize);

	auto test_dataset = CIFAR10(CIFAR10_DATASET_PATH, CIFAR10::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(norm_mean, norm_std))
		.map(torch::data::transforms::Stack<>());

	const size_t test_dataset_size = test_dataset.size().value();

	auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(test_dataset),
																								kTestBatchSize);
	/*std::cout << "typeid train_loader : " << typeid(train_loader).name() << \
			", typeid test_loader : " << typeid(test_loader).name() << std::endl;*/

	float lr = 0.1;
	torch::optim::SGD optimizer(net->parameters(), lr);

	int64_t num_epochs = 100;
	train(net, num_epochs, train_loader, optimizer, device);
	test(net, test_loader, device);
	// torch::save()
	// torch::save(net., "my_net.pt");

	return 0;
}
