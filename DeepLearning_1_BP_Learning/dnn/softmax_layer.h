#pragma once

#include "layer.h"

//-----output-----//
class SoftmaxLayer : public NetworkLayer
{
public:
	void activation(const std::vector<std::vector<float> > &in, std::vector<std::vector<float> > &o_out, bool use_result_to_bp = true) override;

	size_t get_neuron_count() const override;

	// output layer なのでBPによる更新はなし
	void update_weights_by_bp(const std::vector<std::vector<float> > &, float, std::vector<std::vector<float> > &);

	// ロス計算
	float calc_loss(const std::vector<float> &result, size_t gt_index);

	// 微分計算
	void calc_delta(const std::vector<float> &result, const size_t gt_index, std::vector<float> &o_delta);

	// 出力結果から推定されたクラスを取得
	static int calc_estimated_class_index(const std::vector<float> &softmax_output);

};
