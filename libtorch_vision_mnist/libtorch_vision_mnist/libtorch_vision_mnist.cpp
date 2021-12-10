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
#include "dataset.h"
#include "mnist.h"
#include "transform.h"
#include "convnet.h"

using transform::ConstantPad;
using transform::RandomCrop;
using transform::RandomHorizontalFlip;
using transform::RandomVerticalFlip;

int main()
{
	// check cuda available
	std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	// fix seed
	torch::manual_seed(42);
	torch::cuda::manual_seed(42);
	torch::cuda::manual_seed_all(42);

	// Hyper parameters
	const int64_t num_classes = 3;
	const int64_t batch_size = 64;
	const size_t num_epochs = 30;
	const size_t log_interval = 100;
	const double learning_rate = 1e-4; // 1-e4
	const size_t lr_scheduler_stepsize = 2;  // number of epochs after which to decay the learning rate
	const double lr_scheduler_factor = 0.2; // decay factor
	const bool class_imbalance = false;
	const bool mnist_dev_run = true;

	if (mnist_dev_run) {
		int res = test_mnist("./mnist",
			"./pretrained_models/wide_resnet50_2_mnist.pt",
			num_epochs,
			batch_size,
			learning_rate,
			lr_scheduler_stepsize,
			lr_scheduler_factor);

		return 0;
	}

	// read image list using "CSV"
	const std::string file_names_csv = "./train.csv";
	const std::string val_file_names_csv = "./val.csv";

	// declare dataset
	auto train_dataset = CFRPDataset(file_names_csv)
		.map(torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
		.map(RandomHorizontalFlip())
		.map(RandomVerticalFlip())
		.map(RandomCrop({ 224, 224 }))
		.map(torch::data::transforms::Stack<>());

	auto val_dataset = CFRPDataset(val_file_names_csv, CFRPDataset::Mode::kTest)
		.map(torch::data::transforms::Normalize<>({ 0.485, 0.456, 0.406 }, { 0.229, 0.224, 0.225 }))
		.map(RandomCrop({ 224, 224 }))
		.map(torch::data::transforms::Stack<>());

	auto num_train_samples = train_dataset.size().value();
	auto num_val_samples = val_dataset.size().value();
	
	// to solve class imbalance problem
	float* classes_num = CFRPDataset(file_names_csv).count_classes();
	float max_classes_num = *std::max_element(classes_num, classes_num + num_classes);
	for (int i = 0; i < num_classes; i++) {
		classes_num[i] = max_classes_num / classes_num[i];
	}

	// manualy set class imbalance weight
	/* 
	classes_num[0] = 1.;
	classes_num[1] = 3.;
	classes_num[2] = 20;
	*/

	// declare dataloader
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(train_dataset),
		batch_size);

	/*auto val_data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(val_dataset),
		batch_size);*/

	auto val_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
		std::move(val_dataset),
		batch_size);

	// imagenet pretrained model
	const std::string pretrained_classification_model_path = "resnet18_scriptmodule.pt"; // "resnet50.pth";
	// const std::string pretrained_classification_model_path = "swsl_resnext50_32x4d.pth"; 
	torch::jit::script::Module traincls_model;

	// Check if it's loading normally
	try {
		traincls_model = torch::jit::load(pretrained_classification_model_path);
	}
	catch (const torch::Error& error) {
		std::cerr << "Could not load scriptmodule from file " << pretrained_classification_model_path << ".\n";
		return -1;
	}

	/*ConvNet model(3);
	model-*/

	// feature_extractor, fc LR 조정하기
	std::vector<torch::Tensor> feature_extractor;  // Params to apply weight decay
	std::vector<torch::Tensor> fc;  // Params to not apply weight decay
	for (const auto& param : traincls_model.named_parameters()) {
		const auto& name = param.name;
		// fc
		if (name.find("fc") >= 0) {
			fc.push_back(param.value);
		}
		// feature_extractor
		else {
			feature_extractor.push_back(param.value);
		}
	}

	std::vector<torch::optim::OptimizerParamGroup> param_groups{
		// learning_rate * 0.5
		torch::optim::OptimizerParamGroup(
		  feature_extractor,
		  std::make_unique<torch::optim::AdamWOptions>(
			torch::optim::AdamWOptions(learning_rate * 0.5)
		  )
		),
		// learning_rate
		torch::optim::OptimizerParamGroup(
		  fc,
		  std::make_unique<torch::optim::AdamWOptions>(
			torch::optim::AdamWOptions(learning_rate)
		  )
		)
	};

	// declare optimizer and scheduler
	/*for (auto& x : traincls_model.parameters()) {
		std::cout << typeid(x).name() << std::endl;
	}*/

	torch::optim::AdamW optimizer(param_groups, torch::optim::AdamWOptions(learning_rate));
	scheduler::StepLR<decltype(optimizer)> scheduler(optimizer, lr_scheduler_stepsize, lr_scheduler_factor);

	// to Device (CUDA or CPU)
	traincls_model.to(device);

	// fix precision
	std::cout << std::fixed << std::setprecision(4);
	// auto current_learning_rate = learning_rate;
	std::cout << "Training...\n";
	double best_acc = 0.0;

	// training...
	for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
		size_t batch_idx = 0;
		double running_loss = 0.; // mean squared error
		size_t num_correct = 0;

		// mode -> train
		traincls_model.train();
		for (auto& batch : *data_loader) {
			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			/*std::cout << "train... imgs: " << imgs.min() << " " << imgs.max() << std::endl;
			std::cout << labels << '\n';*/ 
			/*std::cout << "imgs: " << imgs.sizes() << std::endl \
				<< "labels: " << labels.sizes() << std::endl;*/

			imgs = imgs.to(torch::kF32).to(device);
			labels = labels.to(torch::kInt64).to(device);

			std::vector<torch::jit::IValue> input;

			input.push_back(imgs);
		
			auto output = traincls_model.forward(input).toTensor();
			
		/*	std::cout << "output: " << output.sizes() << std::endl \
				<< "labels: " << labels.sizes() << std::endl;*/

			at::Tensor loss;
			// apply class_imbalance weight
			if (class_imbalance) { 
				loss = torch::nn::functional::cross_entropy(output,
															labels,
															torch::nn::functional::CrossEntropyFuncOptions().weight(torch::from_blob(classes_num, { 3 }).to(device)));
			}
			else {
				loss = torch::nn::functional::cross_entropy(output, labels);
			}
			
			running_loss += loss.item<double>() * imgs.size(0);

			/*std::cout << "Train ! \n";
			std::cout << output << std::endl;*/

			auto prediction = output.argmax(1);

			num_correct += prediction.eq(labels).sum().item<int64_t>();

			batch_idx++;
			if (batch_idx % log_interval == 0)
			{
				std::printf(
					"\rTrain Epoch: %d/%ld [%5ld/%5d] Loss: %.4f\n",
					epoch,
					num_epochs,
					batch_idx * batch.data.size(0),
					num_train_samples,
					loss.item<double>() * imgs.size(0));
			}

			optimizer.zero_grad();
			loss.backward();
			optimizer.step();
		}

		// print train results
		auto sample_mean_loss = running_loss / num_train_samples;
		auto accuracy = static_cast<double>(num_correct) / num_train_samples;

		std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "], Trainset - Loss: "
			<< sample_mean_loss << ", Accuracy: " << accuracy << '\n';

		std::cout << "Training finished!\n\n";
		std::cout << "Valid sequence...\n";

		// validate(test) the model
		torch::NoGradGuard no_grad;
		traincls_model.eval();

		double val_running_loss = 0.0;
		size_t val_num_correct = 0;

		for (const auto& batch : *val_data_loader) {
			auto imgs = batch.data;
			auto labels = batch.target.squeeze();

			/*std::cout << "valid... imgs: " << imgs.min() << " " << imgs.max() << std::endl;
			std::cout << labels << '\n';*/

			imgs = imgs.to(torch::kF32).to(device);
			labels = labels.to(torch::kInt64).to(device);

			std::vector<torch::jit::IValue> input;
			input.push_back(imgs);

			auto output = traincls_model.forward(input).toTensor();
			auto loss = torch::nn::functional::cross_entropy(output, labels);

			/*std::cout << "Validataion ! \n";
			std::cout << output << std::endl;*/
			val_running_loss += loss.item<double>() * imgs.size(0);

			auto prediction = output.argmax(1);

			val_num_correct += prediction.eq(labels).sum().item<int64_t>();

			/*std::cout << "Valid Pred...........\n";
			std::cout << prediction << '\n';
			
			std::cout << "Valid Label...........\n";
			std::cout << labels << '\n';*/
		}

		std::cout << "Testing finished!\n";

		auto test_accuracy = static_cast<double>(val_num_correct) / num_val_samples;
		auto test_sample_mean_loss = val_running_loss / num_val_samples;

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

		if (test_accuracy > best_acc) {
			best_acc = test_accuracy;
			std::cout << "Saving model" << std::endl;
			traincls_model.save("best_cls.pth");
		}

		// stepLR
		scheduler.step();

		// manual stepLR
		//if ((epoch + 1) % learning_rate_decay_frequency == 0) {
		//	current_learning_rate *= learning_rate_decay_factor;
		//	static_cast<torch::optim::AdamOptions&>(optimizer.param_groups().front()
		//		.options()).lr(current_learning_rate);

		//	std::cout << "Decay Scheduler | current_learning_rate: " << current_learning_rate << std::endl;
		//}
	}

	std::cout << "best test_accuracy: " << best_acc << std::endl;

    return 0;
}