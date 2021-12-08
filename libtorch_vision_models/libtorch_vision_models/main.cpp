// dataset 선언
// model 선언 (모델 다양하게 선택할 수 있도록)
// optimzier, scheduler, loss 선언

#include <iostream>
#include <memory>
#include <torch/torch.h>
#include <torch/nn/pimpl.h>
#include <torchvision/vision.h>
#include <torchvision/models/models.h>
#include <torchvision/models/modelsimpl.h>
#include "cifar10.h"

//auto create_model(const std::string model_type, int64_t num_classes)
void create_model(const std::string model_type, int64_t num_classes, torch::nn::AnyModule &model)
{	
	vision::models::AlexNet         alexnet;
	vision::models::VGG19           vgg19;
	vision::models::ResNet18        resnet18;
	vision::models::InceptionV3     inceptionv3; // 299
	vision::models::MobileNetV2     mobilenet;
	vision::models::ResNext50_32x4d resnext50;
	vision::models::WideResNet50_2  wide_resnet50;

	if (model_type.find("alexnet") != -1) {
		alexnet = vision::models::AlexNet();

		// load imagenet weight
		torch::load(alexnet, model_type);

		// unregister "fc"
		alexnet->unregister_module("classifier");

		// https://github.com/pytorch/vision/blob/main/torchvision/csrc/models/alexnet.cpp
		alexnet->classifier = torch::nn::Sequential(
			torch::nn::Dropout(),
			torch::nn::Linear(256 * 6 * 6, 4096),
			torch::nn::Functional(torch::relu),
			torch::nn::Dropout(),
			torch::nn::Linear(4096, 4096),
			torch::nn::Functional(torch::relu),
			torch::nn::Linear(4096, num_classes)
		);

		// register "fc"
		alexnet->register_module("classifier", alexnet->classifier);
		
		//res = std::make_unique< vision::models::AlexNet::>(alexnet);
			
		// return alexnet;

		model = torch::nn::AnyModule{ alexnet };
	}
	else if (model_type.find("vgg") != -1) {
		vgg19 = vision::models::VGG19();

		// load imagenet weight
		torch::load(vgg19, model_type);

		// unregister "fc"
		vgg19->unregister_module("classifier");

		// https://github.com/pytorch/vision/blob/main/torchvision/csrc/models/vgg.cpp
		vgg19->classifier = torch::nn::Sequential(
			torch::nn::Linear(512 * 7 * 7, 4096),
			torch::nn::Functional(vision::models::modelsimpl::relu_),
			torch::nn::Dropout(),
			torch::nn::Linear(4096, 4096),
			torch::nn::Functional(vision::models::modelsimpl::relu_),
			torch::nn::Dropout(),
			torch::nn::Linear(512, num_classes)
		);

		// register "fc"
		vgg19->register_module("classifier", vgg19->classifier);

		model = torch::nn::AnyModule{ vgg19 };
		// 	return vgg19;
	}
	else if (model_type.find("resnet18") != -1) {
		resnet18 = vision::models::ResNet18();

		// load imagenet weight
		torch::load(resnet18, model_type);

		// unregister "fc"
		resnet18->unregister_module("fc");

		resnet18->fc = torch::nn::Linear(torch::nn::LinearImpl(512, num_classes));

		// register "fc"
		resnet18->register_module("fc", resnet18->fc);

		model = torch::nn::AnyModule{ resnet18 };
	}
	// ToDo (MobileNet)
	else if (model_type.find("mobilenet") != -1) {

	}
	// ToDo (ResNext)
	else if (model_type.find("resnext") != -1) {

	}
	else if (model_type.find("wide_resnet") != -1) {
		wide_resnet50 = vision::models::WideResNet50_2();

		// load imagenet weight
		torch::load(wide_resnet50, model_type);

		// unregister "fc"
		wide_resnet50->unregister_module("fc");

		wide_resnet50->fc = torch::nn::Linear(torch::nn::LinearImpl(2048, num_classes));

		// register "fc"
		wide_resnet50->register_module("fc", wide_resnet50->fc);

		// res = std::make_unique<vision::models::WideResNet50_2>(wide_resnet50);
		model = torch::nn::AnyModule{ wide_resnet50 };
	}
}


template <typename DataLoader>
void test(torch::nn::AnyModule &net,
	DataLoader& test_loader,
	torch::Device device)
{
	net.ptr()->to(device);
	net.ptr()->eval();

	std::cout << "Testing ... " << std::endl;

	size_t num_samples = 0;
	size_t num_correct = 0;
	float running_loss = 0.0;

	for (auto& batch : *test_loader) {
		auto data = batch.data.to(device);
		auto target = batch.target.to(device);

		num_samples += data.size(0);

		torch::Tensor output = net.forward(data);

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
void train(torch::nn::AnyModule& net,
			int64_t num_epochs,
			DataLoader& train_loader,
			torch::optim::Optimizer &optimizer,
			torch::Device device)
{
	net.ptr()->to(device);
	net.ptr()->train();

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

			torch::Tensor output = net.forward(data);

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
	//auto net = vision::models::ResNet18(num_classes);

	//auto net = create_model("pretrained_models/resnet18.pt", 10);
	torch::nn::AnyModule net;
	create_model("pretrained_models/resnet18.pt", 10, net);
	
/*	
	if (dynamic_cast<vision::models::ResNet18*>(*&net))
	{
		std::cout << "error";
	}
	if (dynamic_cast<vision::models::AlexNet*>(&net))
	{
		auto alexNet = static_cast<vision::models::AlexNet*>(&net);
		std::cout << "alexNet";
	}
	*/

	//vision::models::ResNet18 resnet18 = vision::models::ResNet18(num_classes);
	//vision::models::ResNet50 resnet50 = vision::models::ResNet50(num_classes);
	//vision::models::ResNet101 resnet101 = vision::models::ResNet101(num_classes);
	//vision::models::WideResNet50_2 wresnet50_2 = vision::models::WideResNet50_2(num_classes);
	//vision::models::ResNext101_32x8d resnext101_32d8 = vision::models::ResNext101_32x8d(num_classes);

	int64_t kTrainBatchSize = 256;
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
	torch::optim::SGD optimizer(net.ptr()->parameters(), lr);
	
	int64_t num_epochs = 20;
	train(net, num_epochs, train_loader, optimizer, device);
	test(net, test_loader, device);

	torch::save(net.ptr(), "my_net.pt");

	return 0;
}
