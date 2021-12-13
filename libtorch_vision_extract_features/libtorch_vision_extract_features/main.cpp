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
#include "ConfusionMatrix.h"

void create_model(const std::string model_type, int64_t num_classes, std::shared_ptr<torch::nn::AnyModule> &model)
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

		model = std::make_shared<torch::nn::AnyModule>(alexnet);
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

		// model = torch::nn::AnyModule{ vgg19 };
		model = std::make_shared<torch::nn::AnyModule>(vgg19);
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

		// model = torch::nn::AnyModule{ resnet18 };
		// const ModuleHolder<ModuleType>& module_holder
		// module_holder.ptr() == resnet18.ptr();
		// 
		model = std::make_shared<torch::nn::AnyModule>(resnet18);
		
		// std::cout << typeid(resnet18.ptr()).name() << std::endl;
		// torch::nn::ModuleHolder<ModuleType = ResNet18Impl>& module_holder // AnyMoudule.
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
		// model = torch::nn::AnyModule{ wide_resnet50 };
		model = std::make_shared<torch::nn::AnyModule>(wide_resnet50);
	}
}


template <typename DataLoader>
void test(torch::nn::AnyModule &net,
	DataLoader& test_loader,
	torch::Device device,
	ConfusionMatrix &cm)
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

		// 
		for (int i = 0; i < data.size(0); i++) {
			//std::cout << target[i].item<int64_t>() << " " << prediction[i].item<int64_t>() << std::endl;
			cm.accumulate(target[i].item<int64_t>(), prediction[i].item<int64_t>());
		}
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

	for (size_t epoch = 0; epoch < num_epochs; ++epoch)
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

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << std::endl;
	}
}

int main(void)
{
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << std::endl;

	const int num_classes = 10;
	//auto net = vision::models::ResNet18(num_classes);

	//auto net = create_model("pretrained_models/resnet18.pt", 10);
	// torch::nn::AnyModule *net = nullptr; // initialize
	std::shared_ptr<torch::nn::AnyModule> net;

	create_model("pretrained_models/resnet18.pt", num_classes, net);
	
	//vision::models::ResNet18 resnet18 = vision::models::ResNet18(num_classes);
	//vision::models::ResNet50 resnet50 = vision::models::ResNet50(num_classes);
	//vision::models::ResNet101 resnet101 = vision::models::ResNet101(num_classes);
	//vision::models::WideResNet50_2 wresnet50_2 = vision::models::WideResNet50_2(num_classes);
	//vision::models::ResNext101_32x8d resnext101_32d8 = vision::models::ResNext101_32x8d(num_classes);

	//*You can use c++ as follows.
	//vision::models::WideResNet50_2 module_wideresnet_50_;//https://github.com/pytorch/vision
	//auto anomaly_features = torch::jit::load("patchcore_features.pt");
	//anomaly_features.attr("feature").toTensor().to(at::kCUDA);

	//torch::load(module_wideresnet_50_, "patchcore_model.pt");
	//module_wideresnet_50_->eval();
	//module_wideresnet_50_->to(at::kCUDA);

	//auto inputs = get_inputs();//image tensor
	//auto x = module_wideresnet_50_->conv1->forward(inputs);
	//x = module_wideresnet_50_->bn1->forward(x).relu_();
	//x = torch::max_pool2d(x, 3, 2, 1);

	// * instead of register_forward_hook
	// auto outputs1 = module_wideresnet_50_->layer1->forward(x);
	// auto outputs2 = module_wideresnet_50_->layer2->forward(outputs1);
	// auto outputs3 = module_wideresnet_50_->layer3->forward(outputs2);
	
	// 목적: vision::models::ResNet18로 casting 후 안에 정의되어 있는 conv1 뽑아내기

	/*auto ptr = dynamic_cast<torch::nn::Module*>(net->ptr().get());
	if (ptr == nullptr) {
		std::cout << "nullptr" << std::endl;
	}*/
	
	auto net2visions = std::dynamic_pointer_cast<struct vision::models::ResNet18Impl>(net->ptr());

	if (net2visions == nullptr) {
		std::cout << "net2visions || dynamic cast fail" << std::endl;
	}

	std::cout << typeid(net2visions).name() << std::endl;

	auto sample_tensor = torch::randn({ 1, 3, 224, 224 });

	auto x = net2visions->conv1->forward(sample_tensor);
	x = net2visions->bn1->forward(x).relu_();
	x = torch::max_pool2d(x, 3, 2, 1);
	auto output1 = net2visions->layer1->forward(x);
	auto output2 = net2visions->layer2->forward(output1);
	auto output3 = net2visions->layer3->forward(output2);

	std::cout << "output1 sizes: " << output1.sizes() << std::endl;
	std::cout << "output2 sizes: " << output2.sizes() << std::endl;
	std::cout << "output3 sizes: " << output3.sizes() << std::endl;

#if 0
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

	float lr = 0.1;
	torch::optim::SGD optimizer(net.ptr()->parameters(), lr);
	
	ConfusionMatrix /*traincm(num_classes),*/ testcm(num_classes);

	int64_t num_epochs = 10;
	train(net, num_epochs, train_loader, optimizer, device);
	test(net, test_loader, device, testcm);

	std::cout << "accuracy: " << testcm.accuracy() << std::endl;
	std::cout << "average precision: " << testcm.avgPrecision() << std::endl;
	std::cout << "average recall: " << testcm.avgRecall() << std::endl;
	
	testcm.printCounts();

	torch::save(net.ptr(), "my_net.pt");
#endif
	return 0;
}
