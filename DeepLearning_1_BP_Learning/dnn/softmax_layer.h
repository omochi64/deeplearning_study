#pragma once

#include "layer.h"

//-----output-----//
class SoftmaxLayer : public NetworkLayer
{
public:
	void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true) override;

	size_t get_neuron_count() const override;

	// output layer �Ȃ̂�BP�ɂ��X�V�͂Ȃ�
	void update_weights_by_bp(const std::vector<std::vector<double> > &, double, std::vector<std::vector<double> > &);

	// ���X�v�Z
	double calc_loss(const std::vector<double> &result, size_t gt_index);

	// �����v�Z
	void calc_delta(const std::vector<double> &result, const size_t gt_index, std::vector<double> &o_delta);

	// �o�͌��ʂ��琄�肳�ꂽ�N���X���擾
	static int calc_estimated_class_index(const std::vector<double> &softmax_output);

};
