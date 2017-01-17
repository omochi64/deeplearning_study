
#include <vector>
#include <cassert>
#include "softmax_layer.h"

void SoftmaxLayer::activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp)
{
	o_out.clear();
	size_t dim = in[0].size();
	o_out.resize(in.size(), std::vector<double>(dim));

	for (size_t data_i = 0; data_i < in.size(); data_i++)
	{
		auto &in_i = in[data_i];
		double sum_of_exp = 0;
		for (auto i = 0; i < in_i.size(); i++)
		{
			double v = in_i[i];
			auto exp_v = std::exp(v);
			sum_of_exp += exp_v;
			o_out[data_i][i] = exp_v;
		}
		for (auto i = 0; i < in_i.size(); i++)
		{
			o_out[data_i][i] /= sum_of_exp;
		}
	}
}


size_t SoftmaxLayer::get_neuron_count() const
{
	return 1;
}

int SoftmaxLayer::calc_estimated_class_index(const std::vector<double> &softmax_output)
{
	double max = -199999999;
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
void SoftmaxLayer::update_weights_by_bp(const std::vector<std::vector<double> > &, double, std::vector<std::vector<double> > &)
{
	assert(false);
}

// ロス計算
double SoftmaxLayer::calc_loss(const std::vector<double> &result, size_t gt_index)
{
	return -std::log(result[gt_index]); // gt_index以外の項は0になる

}

// 微分計算
void SoftmaxLayer::calc_delta(const std::vector<double> &result, const size_t gt_index, std::vector<double> &o_delta)
{
	assert(gt_index >= 0 && gt_index < result.size());

	o_delta.resize(result.size());

	for (size_t i = 0; i < result.size(); i++)
	{
		o_delta[i] = result[i] - (i == gt_index ? 1 : 0);
	}
}
