#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <ctime>
#include <thread>

#include "data.h"
#include "pgm.h"
#include "iris_loader.h"
#include "mnist_loader.h"
#include "dnn/activations.h"
#include "dnn/layer.h"
#include "dnn/fully_connected_layer.h"
#include "dnn/softmax_layer.h"
#include "simple_timer.h"




int main(int argc, char *argv[])
{
	std::string fname = "iris_data_dummy.csv";

	srand(0);

	//std::vector<Instance> iris_dataset;
	//load_iris_dataset(fname, iris_dataset, true);

	std::vector<Instance> dataset;
	std::vector<Instance> testset;

	bool is_mnist = true;

	if (is_mnist)
	{
		load_mnist_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte", dataset);
		load_mnist_dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", testset);
	}
	else
	{
		load_iris_dataset(fname, dataset, true);
		load_iris_dataset(fname, testset, true);
	}
	

	// input - 5(+1) - 4(+1) - labels - softmax
	auto activ = std::make_shared<ReLUActivation>();

	std::shared_ptr<FullConnectedLayer> first_layer;
	first_layer = std::make_shared<FullConnectedLayer>(10, (unsigned int)(dataset[0].data.size()+1), activ, false);
	//auto second = std::make_shared<FullConnectedLayer>(150, first_layer->get_neuron_count(), activ, false);
	//first_layer->set_next(second);
	//auto third = std::make_shared<FullConnectedLayer>(10, first_layer->get_neuron_count(), activ, false);
	//first_layer->set_next(third);
	auto softmax = std::make_shared<SoftmaxLayer>();
	first_layer->set_next(softmax);

	int batch_size = 100;
	float bias_init = 0.1;

	// update by bp
	for (size_t bp = 0; bp < 1000; bp++)
	{
		std::cout << bp << "th update" << std::endl;
		
		unsigned int correct_count = 0;

		float time_for_forward = 0;
		float time_for_bp = 0;

		float time_for_calc_delta = 0;
		float time_for_calc_loss = 0;
		float time_for_update = 0;

		for (int i = 0; i < dataset.size();)
		{
			// feed forward
			std::vector<std::vector<float> > input_array; input_array.reserve(batch_size);
			std::vector<unsigned int> label_array;
			for (int batch_i = 0; batch_i < batch_size; batch_i++)
			{
				if (batch_i + i >= dataset.size()) continue;
				auto input_data = dataset[batch_i + i].data;
				input_data.push_back(bias_init);
				input_array.push_back(input_data);
				label_array.push_back(dataset[batch_i+i].cls);
			}
			
			std::vector<std::vector<float> > output_array;

			SimpleTimer timer_forward;

			first_layer->activation_chain(input_array, output_array);

			// get the predicted class
			std::vector<std::vector<float> > loss_delta(output_array.size());
			for (size_t data_i = 0; data_i < output_array.size(); data_i++)
			{
				int max_idx = SoftmaxLayer::calc_estimated_class_index(output_array[data_i]);

				//std::cout << i << ": " << iris.cls << " est:" << max_idx << " ," << (iris.cls == max_idx ? "correct" : "wrong") << std::endl;
				if (label_array[data_i] == max_idx) correct_count++;

				softmax->calc_delta(output_array[data_i], label_array[data_i], loss_delta[data_i]);
			}
			time_for_forward += timer_forward.past_seconds();

			SimpleTimer timer_bp;

			// backprop
			softmax->update_bp_from_output_layer(loss_delta, 0.5);

			time_for_bp += timer_bp.past_seconds();

			time_for_calc_delta += /*third->calc_delta_past_sec_*/ + first_layer->calc_delta_past_sec_;
			time_for_calc_loss += /*third->calc_bp_loss_past_sec_*/ + first_layer->calc_bp_loss_past_sec_;
			time_for_update += /*third->update_weights_past_sec_*/ + first_layer->update_weights_past_sec_;

			int next_i = i + batch_size;
			if      (dataset.size() * 3 / 4 >= i && dataset.size() * 3 / 4 < next_i) std::cout << "75% of epoch finished" << std::endl;
			else if (dataset.size() * 2 / 4 >= i && dataset.size() * 2 / 4 < next_i) std::cout << "50% of epoch finished" << std::endl;
			else if (dataset.size() * 1 / 4 >= i && dataset.size() * 1 / 4 < next_i) std::cout << "25% of epoch finished" << std::endl;

			i = next_i;
		}

		std::cout << "Time for forward, bp = " << time_for_forward << "," << time_for_bp << std::endl;
		std::cout << "Time for calc delta, loss, update = " << time_for_calc_delta << "," << time_for_calc_loss << "," << time_for_update << std::endl;

		std::cout << "correct rate (training): " << correct_count << "/" << dataset.size() << " = "  << (100.0 * correct_count / dataset.size()) << "%" << std::endl;

		if (true)
		{
			float loss_total = 0;
			int correct_total = 0;
			std::cout << "calculating test loss..." << std::endl;
			for (size_t i = 0; i < testset.size(); i++)
			{
				auto &data = testset[i];
				std::vector<float> input(data.data);
				input.push_back(bias_init);  // bias term
				std::vector<std::vector<float> > input_array(1); input_array[0] = input;
				std::vector<std::vector<float> > output_array;

				first_layer->activation_chain(input_array, output_array);
				int max_idx = SoftmaxLayer::calc_estimated_class_index(output_array[0]);
				if (max_idx == data.cls) correct_total++;

				loss_total += softmax->calc_loss(output_array[0], data.cls);
			}
			std::cout << "test loss (total): " << loss_total << std::endl;
			std::cout << "test correct rate: " << correct_total << "/" << testset.size() << " = " << (100.0f * correct_total / testset.size()) << "%" << std::endl;
		}
	}

	

	return 0;
}
