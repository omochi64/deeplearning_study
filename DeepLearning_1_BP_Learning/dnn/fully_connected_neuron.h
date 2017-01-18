#pragma once

#include <vector>
#include <memory>

class ActivationAbst;

// ニューロン1個
class FullConnectedNeuron
{
public:
	explicit FullConnectedNeuron(unsigned int input_count, std::shared_ptr<ActivationAbst> activator);

	void activation(const std::vector<std::vector<float> > &in, std::vector<float> &o_out, bool save_activ_deriv = true);

	// dIn_this/dOut_prev: 左記の値を返す。FullConnectedの場合は重みに相当
	inline float calc_deriv_btw_in_out(int in_index) const
	{
		assert(in_index >= 0 && in_index < weights_.size());

		return weights_[in_index];
	}

	// dIn_this/dOut_prev: 左記の値をまとめて返す。FullConnectedの場合は重みに相当
	inline const std::vector<float> get_deriv_btw_in_out_array() const
	{
		return weights_;
	}

	// 最新の活性化関数微分値を返す
	inline const std::vector<float> &get_latest_activ_deriv() const
	{
		return latest_activ_deriv_;
	}

	// 与えられたdeltaに基づいて更新
	void update_weights(const std::vector<std::vector<float> > &in_from_prev_layer, const std::vector<float> &delta, float learning_rate);
	
	// 与えられたdiffに基づいて更新
	void update_weights(const float *minus_diff);

	void set_weights_directly(const float *weights);

	void print_weights();

private:
	static float adjust_weight(float weight);

private:
	std::vector<float> weights_;
	std::vector<float>  latest_activ_deriv_;
	std::shared_ptr<ActivationAbst> activator_;
};
