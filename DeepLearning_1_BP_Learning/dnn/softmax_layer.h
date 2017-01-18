#pragma once

#include "layer.h"

//-----output-----//
class SoftmaxLayer : public NetworkLayer
{
public:
	void activation(const std::vector<std::vector<float> > &in, std::vector<std::vector<float> > &o_out, bool use_result_to_bp = true) override;

	size_t get_neuron_count() const override;

	// output layer �Ȃ̂�BP�ɂ��X�V�͂Ȃ�
	void update_weights_by_bp(const std::vector<std::vector<float> > &, float, std::vector<std::vector<float> > &);

	// ���X�v�Z
	float calc_loss(const std::vector<float> &result, size_t gt_index);

	// �����v�Z
	void calc_delta(const std::vector<float> &result, const size_t gt_index, std::vector<float> &o_delta);

	// �o�͌��ʂ��琄�肳�ꂽ�N���X���擾
	static int calc_estimated_class_index(const std::vector<float> &softmax_output);

};
