#pragma once

#include <vector>
#include <memory>

class ActivationAbst;

// �j���[����1��
class FullConnectedNeuron
{
public:
	explicit FullConnectedNeuron(unsigned int input_count, std::shared_ptr<ActivationAbst> activator);

	void activation(const std::vector<std::vector<double> > &in, std::vector<double> &o_out, bool save_activ_deriv = true);

	// dIn_this/dOut_prev: ���L�̒l��Ԃ��BFullConnected�̏ꍇ�͏d�݂ɑ���
	inline double calc_deriv_btw_in_out(int in_index) const
	{
		assert(in_index >= 0 && in_index < weights_.size());

		return weights_[in_index];
	}

	// �ŐV�̊������֐������l��Ԃ�
	inline const std::vector<double> &get_latest_activ_deriv() const
	{
		return latest_activ_deriv_;
	}

	// �^����ꂽdelta�Ɋ�Â��čX�V
	void update_weights(const std::vector<std::vector<double> > &in_from_prev_layer, const std::vector<double> &delta, double learning_rate);

	void print_weights();

private:
	std::vector<double> weights_;
	std::vector<double>  latest_activ_deriv_;
	std::shared_ptr<ActivationAbst> activator_;
};
