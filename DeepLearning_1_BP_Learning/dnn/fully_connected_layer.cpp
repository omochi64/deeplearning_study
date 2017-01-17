#include <assert.h>
#include <iostream>

#include "fully_connected_layer.h"
#include "activations.h"
#include "simple_timer.h"




//-----HiddenLayer(FC)-----//
FullConnectedLayer::FullConnectedLayer(unsigned int neuron_count, unsigned int prev_layer_output_count, std::shared_ptr<ActivationAbst> activator, bool add_bias_term)
	: NetworkLayer()
	, add_bias_term_(add_bias_term)
	, prev_layer_output_count_(prev_layer_output_count)
{
	neurons_.clear();
	neurons_.reserve(neuron_count);

	for (size_t i = 0; i < neuron_count; i++)
	{
		neurons_.emplace_back(prev_layer_output_count, activator);
	}
}

void FullConnectedLayer::activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp)
{
	if (use_result_to_bp)
	{
		latest_input_ = in;
	}

	size_t data_count = in.size();
	o_out.clear();
	o_out.resize(data_count);

	size_t neuron_count = neurons_.size();
	auto *neurons_array = neurons_.data();

	std::vector<double> *out_array = o_out.data();

	for (size_t data_i = 0; data_i < data_count; data_i++)
	{
		o_out[data_i].resize(neuron_count);
		if (add_bias_term_)
		{
			// add_bias_term が true の場合、出力にBiasとなる「1」だけを出力するニューロンを追加する
			// *入力にbiasを入れるのではなく、この層がbiasを出力する
			o_out[data_i].push_back(1);
		}
	}

	//#pragma omp parallel
	for (size_t i = 0; i < neuron_count; i++)
	{
		auto &neuron = neurons_array[i];

		std::vector<double> activated_values;
		neuron.activation(in, activated_values, use_result_to_bp);

		for (size_t data_i = 0; data_i < data_count; data_i++)
		{
			o_out[data_i][i] = activated_values[data_i];
		}
	}
}

	// in_deriv_by_out_x_delta_of_next_layer: list of [d(In_nextLayer)/d(Out_thisLayer) * delta_nextLayer]: size = neuron_count_of_this_layer
void FullConnectedLayer::update_weights_by_bp(const std::vector<std::vector<double> > &in_deriv_out_x_delta_of_next_layer,
	double learning_rate,
	std::vector<std::vector<double> > &o_in_deriv_out_x_delta_of_this_layer)
{
	assert(in_deriv_out_x_delta_of_next_layer[0].size() == get_neuron_count());

	std::vector<std::vector<double> > delta;
	delta.resize(in_deriv_out_x_delta_of_next_layer.size());

	auto *neurons_ptr = neurons_.data();
	std::vector<double> *delta_ptr = delta.data();
	auto *in_deriv_ptr = in_deriv_out_x_delta_of_next_layer.data();

	SimpleTimer delta_timer;

	// この層のニューロンごとのdeltaを求める
	// ニューロンのdelta = 活性化関数の微分値(ニューロンへの入力値) * 前段からの計算値
	for (size_t data_i = 0; data_i < delta.size(); data_i++)
	{
		delta[data_i].resize(get_neuron_count_wo_bias());
		for (size_t i = 0; i < get_neuron_count_wo_bias(); i++)
		{
			double delta_i = neurons_ptr[i].get_latest_activ_deriv()[data_i] * in_deriv_ptr[data_i][i];
			delta[data_i][i] = delta_i;
		}
	}

	calc_delta_past_sec_ = delta_timer.past_seconds();

	SimpleTimer calc_bp_loss_timer;

	// 次のBP用に「d この層への入力(activation直前)_per_this_layer_neuron/ d 前の層の出力_per_prev_layer_neuron * delta」を求める
	// 「前段からの計算値」に相当
	// ↑の値は、this layer neuron 全てに対して和をとる
	// ↑の微分は、FullyConnectedの場合はニューロン間の重みになる
	//std::vector<double> next_bp_output;  // サイズは prev_layer_output_count_ になる
	//next_bp_output.resize(prev_layer_output_count_);
	o_in_deriv_out_x_delta_of_this_layer.resize(delta.size());
	//o_in_deriv_out_x_delta_of_this_layer.resize(prev_layer_output_count_);
	auto *next_bp_ptr = o_in_deriv_out_x_delta_of_this_layer.data();

	for (size_t data_i = 0; data_i < delta.size(); data_i++)
	{
		next_bp_ptr[data_i].resize(prev_layer_output_count_);
	}

#pragma omp parallel
	for (size_t data_i = 0; data_i < delta.size(); data_i++)
	{
		for (size_t i = 0; i < prev_layer_output_count_; i++)
		{
			// この層の全てのニューロンに対して合計を取る
			double value = 0;
			for (size_t k = 0; k < get_neuron_count_wo_bias(); k++)
			{
				// d この層への入力(activation直前)_per_this_layer_neuron/ d 前の層の出力_per_prev_layer_neuron: 重みに相当
				auto in_deriv_out = neurons_ptr[k].calc_deriv_btw_in_out(i);

				value += in_deriv_out * delta[data_i][k];
			}

			next_bp_ptr[data_i][i] = value;
		}
	}
	calc_bp_loss_past_sec_ = calc_bp_loss_timer.past_seconds();

	SimpleTimer update_timer;

	// 計算したdeltaを用いて更新
	// W_ij_next = W_ij_cur - learning_rate * delta_j * Out_i
	size_t neurons_count = get_neuron_count_wo_bias();
#pragma omp parallel
	for (size_t j = 0; j < neurons_count; j++)
	{
		auto &neuron = neurons_ptr[j];
		std::vector<double> delta_for_j;
		for (size_t data_i = 0; data_i < delta.size(); data_i++) delta_for_j.push_back(delta[data_i][j]);
		neuron.update_weights(latest_input_, delta_for_j, learning_rate);
	}

	update_weights_past_sec_ = update_timer.past_seconds();
}

void FullConnectedLayer::print_weights()
{
	{
		for (auto &neuron : neurons_)
		{
			neuron.print_weights();
		}
	}
}