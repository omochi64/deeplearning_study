#pragma once

#include <vector>
#include <memory>

class ActivationAbst;

// ニューロン1個
class FullConnectedNeuron
{
public:
	explicit FullConnectedNeuron(unsigned int input_count, std::shared_ptr<ActivationAbst> activator);

	void activation(const std::vector<std::vector<double> > &in, std::vector<double> &o_out, bool save_activ_deriv = true);

	// dIn_this/dOut_prev: 左記の値を返す。FullConnectedの場合は重みに相当
	inline double calc_deriv_btw_in_out(int in_index) const
	{
		assert(in_index >= 0 && in_index < weights_.size());

		return weights_[in_index];
	}

	// 最新の活性化関数微分値を返す
	inline const std::vector<double> &get_latest_activ_deriv() const
	{
		return latest_activ_deriv_;
	}

	// 与えられたdeltaに基づいて更新
	void update_weights(const std::vector<std::vector<double> > &in_from_prev_layer, const std::vector<double> &delta, double learning_rate);

	void print_weights();

private:
	std::vector<double> weights_;
	std::vector<double>  latest_activ_deriv_;
	std::shared_ptr<ActivationAbst> activator_;
};
