
#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>
#include "softmax_layer.h"

void SoftmaxLayer::activation(const std::vector<std::vector<float> > &in, std::vector<std::vector<float> > &o_out, bool use_result_to_bp)
{
	o_out.clear();
	size_t dim = in[0].size();
	o_out.resize(in.size(), std::vector<float>(dim));

	// 1つでも極端にでかい値があるとexpとった段階でオーバーフローする
	// exp(a) / sum_k(exp(k))
	// = 1/exp(max_v) * exp(a) / sum_k(exp(k) / exp(max_v))
	// = exp(a-max_v) / sum_k(exp(k-max_v))

	for (size_t data_i = 0; data_i < in.size(); data_i++)
	{
		auto &in_i = in[data_i];

		auto max_v = *std::max_element(in_i.begin(), in_i.end());

		float sum_of_exp = 0;
		std::vector<float> orig = in_i;
		for (auto i = 0; i < in_i.size(); i++)
		{
			float v = in_i[i];

#ifdef _DEBUG
			if (std::isnan(v) || std::isinf(v))
			{
				std::cout << "data[" << data_i << "][" << i << "] is nan!" << std::endl;
			}
#endif
			auto exp_v = std::exp(v-max_v);

#ifdef _DEBUG
			if (std::isnan(exp_v) || std::isinf(exp_v))
			{
				std::cout << "exp(data[" << data_i << "][" << i << "] = " << v- max_v << ") is nan!" << std::endl;
			}
#endif
			sum_of_exp += exp_v;
			o_out[data_i][i] = exp_v;
		}
		for (auto i = 0; i < in_i.size(); i++)
		{
			auto prev = o_out[data_i][i];
			o_out[data_i][i] /= sum_of_exp;

#ifdef _DEBUG
			if (std::isnan(o_out[data_i][i]) || std::isinf(o_out[data_i][i]))
			{
				std::cout << "o_out[" << data_i << "][" << i << "] = " << o_out[data_i][i] << " = " << prev << "/" << sum_of_exp << " is nan!" << std::endl;
				std::cout << "input value = " << orig[i] << ", max_v = " << max_v << std::endl;
			}
			//std::cout << o_out[data_i][i] << ",";
#endif // _DEBUG
		}
		//std::cout << std::endl;
	}
}


size_t SoftmaxLayer::get_neuron_count() const
{
	return 1;
}

int SoftmaxLayer::calc_estimated_class_index(const std::vector<float> &softmax_output)
{
	float max = -199999999;
	int max_idx = -1;
	for (int j = 0; j < softmax_output.size(); j++)
	{
		if (max < softmax_output[j])
		{
			max_idx = j;
			max = softmax_output[j];
		}
	}

	return max_idx;
}

// output layer なのでBPによる更新はなし
void SoftmaxLayer::update_weights_by_bp(const std::vector<std::vector<float> > &, float, std::vector<std::vector<float> > &)
{
	assert(false);
}

// ロス計算
float SoftmaxLayer::calc_loss(const std::vector<float> &result, size_t gt_index)
{
	auto value = result[gt_index];
	auto loss = -std::log(value); // gt_index以外の項は0になる
#ifdef _DEBUG
	if (std::isnan(loss) || std::isinf(loss))
	{
		std::cerr << "loss is nan or inf: orig=" << value << ",v=" << loss << std::endl;
	}
#endif
	return loss;
}

// 微分計算
void SoftmaxLayer::calc_delta(const std::vector<float> &result, const size_t gt_index, std::vector<float> &o_delta)
{
	assert(gt_index >= 0 && gt_index < result.size());

	o_delta.resize(result.size());

	for (size_t i = 0; i < result.size(); i++)
	{
		o_delta[i] = result[i] - (i == gt_index ? 1 : 0);

		if (fabs(o_delta[i]) > 1) std::cerr << "loss of softmax exceeds [-1,1] = " << o_delta[i] << ", result=" << result[i] << std::endl;
	}
}
