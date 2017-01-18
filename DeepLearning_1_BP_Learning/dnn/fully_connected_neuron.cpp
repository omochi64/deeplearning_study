#include <vector>
#include <cassert>
#include <iostream>

#include "fully_connected_neuron.h"
#include "activations.h"


// ニューロン1個
FullConnectedNeuron::FullConnectedNeuron(unsigned int input_count, std::shared_ptr<ActivationAbst> activator)
	: activator_(activator)
{
	weights_.resize(input_count);
	for (size_t i = 0; i < input_count; i++) weights_[i] = 0;

	// random init
	for (size_t i = 0; i < input_count; i++)
	{
		weights_[i] = ((rand() % 1001) / 1000.0f - 0.5) * 2;
	}
}

void FullConnectedNeuron::activation(const std::vector<std::vector<float> > &in, std::vector<float> &o_out, bool save_activ_deriv)
{
	assert(in[0].size() == weights_.size());
	o_out.resize(in.size());

	if (save_activ_deriv && latest_activ_deriv_.size() != in.size()) latest_activ_deriv_.resize(in.size());

	for (size_t data_i = 0; data_i < in.size(); data_i++)
	{
		float sum = 0;
		for (size_t i = 0; i < weights_.size(); i++)
		{
			sum += weights_[i] * in[data_i][i];
		}

		if (save_activ_deriv)
		{
			latest_activ_deriv_[data_i] = activator_->calc_deriv(sum);
		}

		o_out[data_i] = activator_->activation(sum);
	}

}


// 与えられたdeltaに基づいて更新
void FullConnectedNeuron::update_weights(const std::vector<std::vector<float> > &in_from_prev_layer, const std::vector<float> &delta, float learning_rate)
{
	assert(in_from_prev_layer[0].size() == weights_.size());

	size_t size = weights_.size();
	float *weights = weights_.data();

	for (size_t i = 0; i < size; i++)
	{
		float error_sum = 0;

		for (size_t data_i = 0; data_i < in_from_prev_layer.size(); data_i++)
		{
			error_sum += in_from_prev_layer[data_i][i] * delta[data_i];
		}

		weights[i] = adjust_weight(weights[i] - learning_rate * error_sum / in_from_prev_layer.size());
	}
}

// 与えられたdiffに基づいて更新
void FullConnectedNeuron::update_weights(const float *minus_diff)
{
	for (size_t i = 0; i < weights_.size(); i++)
	{
		weights_[i] = adjust_weight(weights_[i] - minus_diff[i]);
	}
}

void FullConnectedNeuron::set_weights_directly(const float *weights)
{
	for (size_t i = 0; i < weights_.size(); i++)
	{
		weights_[i] = adjust_weight(weights[i]);
	}
}

float FullConnectedNeuron::adjust_weight(float weight)
{
	if (fabs(weight) > 10)
	{
		return 10 * (weight > 0 ? 1 : -1);
	}
	return weight;
}

void FullConnectedNeuron::print_weights()
{
	std::cout << "[";
	for (auto weight : weights_)
	{
		printf("%.2f ", weight);
	}
	std::cout << "]" << std::endl;
}
