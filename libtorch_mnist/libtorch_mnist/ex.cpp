#include <iostream>
#include <torch/torch.h>

#include <cstddef>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

int main()
{
	std::cout << "---- BASIC AUTOGRAD EXAMPLE 1 ----\n";

	// Create Tensors
	/*torch::Tensor x = torch::tensor({ 1.0 }, torch::requires_grad());
	torch::Tensor w = torch::tensor(2.0, torch::requires_grad());
	torch::Tensor b = torch::tensor(3.0, torch::requires_grad());

	std::cout << x << '\n';
	std::cout << w << '\n';
	std::cout << b << '\n';

	torch::Tensor tensor = torch::ones(5);

	std::cout << tensor << '\n';*/

	torch::Tensor a = torch::tensor({ 1.25345, 2.23452345 }, torch::TensorOptions().dtype(torch::kFloat32));

	std::cout << a << '\n';

	auto k =1 - ( (a) / (torch::sum(a)));

	std::cout <<k  << '\n';


	return 0;
}

