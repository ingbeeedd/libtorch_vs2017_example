#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <torchvision/vision.h>
#include <torchvision/models/resnet.h>

int test_mnist(std::string mnist_path,
				std::string pretrained_classification_model_path,
				size_t num_epochs,
				size_t batch_size,
				double learning_rate,
				size_t lr_scheduler_stepsize,
				double lr_scheduler_factor
);

