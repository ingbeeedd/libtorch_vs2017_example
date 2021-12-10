#include "mnist.h"
#include "scheduler.h"
#include "convnet.h"

int test_mnist(std::string mnist_path,
    std::string pretrained_classification_model_path,
    size_t num_epochs,
    size_t batch_size,
	double learning_rate,
    size_t lr_scheduler_stepsize,
	double lr_scheduler_factor
)
{
	const size_t log_interval = 100;
	const int num_class = 10;
	const bool class_imbalance = false;
	const bool control_lr = false;

    // check cuda available
    std::cout << "CUDA DEVICE COUNT: " << torch::cuda::device_count() << std::endl;
    auto cuda_available = torch::cuda::is_available();
    torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
    std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

    const std::string MNIST_data_path = std::move(mnist_path);

    // MNIST dataset
    auto train_dataset = torch::data::datasets::MNIST(MNIST_data_path)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the training set
    auto num_train_samples = train_dataset.size().value();

    auto test_dataset = torch::data::datasets::MNIST(MNIST_data_path, torch::data::datasets::MNIST::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());

    // Number of samples in the testset
    auto num_test_samples = test_dataset.size().value();

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), batch_size);

    auto test_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
        std::move(test_dataset), batch_size);

 //   torch::jit::script::Module model;
	//// ConvNet model2(10);

 //   // Check if it's loading normally
 //   try {
 //       model = torch::jit::load(std::move(pretrained_classification_model_path));
 //   }
 //   catch (const torch::Error& error) {
 //       std::cerr << "Could not load scriptmodule from file " << pretrained_classification_model_path << ".\n";
 //       return -1;
 //   }


    // feature_extractor, fc LR 조정하기
    //std::vector<torch::Tensor> feature_extractor;  // Params to apply weight decay
    //std::vector<torch::Tensor> fc;  // Params to not apply weight decay
    //for (const auto& param : model.named_parameters()) {
    //    const auto& name = param.name;
    //    // fc
    //    if (name.find("fc") >= 0) {
    //        fc.push_back(param.value);
    //    }
    //    // feature_extractor
    //    else {
    //        feature_extractor.push_back(param.value);
    //    }
    //}

    //std::vector<torch::optim::OptimizerParamGroup> param_groups{
    //    // learning_rate * 0.5
    //    torch::optim::OptimizerParamGroup(
    //      feature_extractor,
    //      std::make_unique<torch::optim::AdamWOptions>(
    //        torch::optim::AdamWOptions(learning_rate * 0.5)
    //      )
    //    ),
    //    // learning_rate
    //    torch::optim::OptimizerParamGroup(
    //      fc,
    //      std::make_unique<torch::optim::AdamWOptions>(
    //        torch::optim::AdamWOptions(learning_rate)
    //      )
    //    )
    //};

	// transfer learning
	std::cout << vision::cuda_version() << std::endl;

	auto model = vision::models::WideResNet50_2();
	
	torch::load(model, pretrained_classification_model_path);

	model->unregister_module("fc");

	model->fc = torch::nn::Linear(torch::nn::LinearImpl(2048, num_class));

	model->register_module("fc", model->fc);


	/*std::vector<at::Tensor> param_groups;
	if (!control_lr) {
		for (const auto& params : model->parameters()) {
			param_groups.emplace_back(params);
		}
	}*/

    // declare optimizer and scheduler
    torch::optim::AdamW optimizer(model->parameters(), torch::optim::AdamWOptions(learning_rate));
	// torch::optim::AdamW optimizer(model2->parameters(), torch::optim::AdamWOptions(learning_rate));
    scheduler::StepLR<decltype(optimizer)> scheduler(optimizer, lr_scheduler_stepsize, lr_scheduler_factor);

    // to Device (CUDA or CPU)
    model->to(device);

	std::cout << model->parameters()[0].device() << std::endl;
	// model2->to(device);

    // fix precision
    std::cout << std::fixed << std::setprecision(4);
    // auto current_learning_rate = learning_rate;
    std::cout << "Training...\n";
    double best_acc = 0.0;
#if 1
	// training...
	for (size_t epoch = 0; epoch != num_epochs; ++epoch) {
		size_t batch_idx = 0;
		double running_loss = 0.; // mean squared error
		size_t num_correct = 0;

		// mode -> train
		model->train();
		// model2->train();
		for (auto& batch : *train_loader) {
			auto imgs = batch.data.to(device);
			auto labels = batch.target.squeeze().to(device);

			// 1-channel => 3-channel
			imgs = torch::cat({ imgs, imgs, imgs }, 1);

			/*std::cout << " imgs sizes: " << imgs.sizes() << std::endl;*/

			// std::cout << imgs.min() << " " << imgs.max() << std::endl;
 
			// std::vector<torch::jit::IValue> input;

			// input.push_back(imgs);
			
			auto output = model->forward(imgs);

			// auto output = model2->forward(imgs);

			at::Tensor loss;
			// apply class_imbalance weight
			if (class_imbalance) {
				loss = torch::nn::functional::cross_entropy(output,
					labels
					// torch::nn::functional::CrossEntropyFuncOptions().weight(torch::from_blob(classes_num, { 3 }).to(device))
				);
			}
			else {
				loss = torch::nn::functional::cross_entropy(output, labels);
			}

			running_loss += loss.item<double>() * imgs.size(0);

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
		model->eval();
		// model2->eval();

		double val_running_loss = 0.0;
		size_t val_num_correct = 0;

		for (const auto& batch : *test_loader) {
			auto imgs = batch.data.to(device);
			auto labels = batch.target.squeeze().to(device);

			imgs = torch::cat({ imgs, imgs, imgs }, 1);

			/*std::vector<torch::jit::IValue> input;

			input.push_back(imgs);*/

			auto output = model->forward(imgs);
			// auto output = model2->forward(imgs);
			auto loss = torch::nn::functional::cross_entropy(output, labels);

			/*std::cout << "Validataion ! \n";
			std::cout << output << std::endl;*/
			val_running_loss += loss.item<double>() * imgs.size(0);

			auto prediction = output.argmax(1);

			val_num_correct += prediction.eq(labels).sum().item<int64_t>();
		}

		std::cout << "Testing finished!\n";

		// std::cout << static_cast<double>(val_num_correct) << " " << num_val_samples << "\n";
		auto test_accuracy = static_cast<double>(val_num_correct) / num_test_samples;
		auto test_sample_mean_loss = val_running_loss / num_test_samples;

		std::cout << "Testset - Loss: " << test_sample_mean_loss << ", Accuracy: " << test_accuracy << '\n';

		if (test_accuracy > best_acc) {
			best_acc = test_accuracy;
			std::cout << "Saving model" << std::endl;
			// model.save("best_cls.pth");
		}

		// stepLR
		scheduler.step();

	}

	std::cout << "best test_accuracy: " << best_acc << std::endl;

#endif
	return 0;
}