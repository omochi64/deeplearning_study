#pragma once

#include "layer.h"

//-----output-----//
class SoftmaxLayer : public NetworkLayer
{
public:
	void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true) override;

	size_t get_neuron_count() const override;

	// output layer なのでBPによる更新はなし
	void update_weights_by_bp(const std::vector<std::vector<double> > &, double, std::vector<std::vector<double> > &);

	// ロス計算
	double calc_loss(const std::vector<double> &result, size_t gt_index);

	// 微分計算
	void calc_delta(const std::vector<double> &result, const size_t gt_index, std::vector<double> &o_delta);

	// 出力結果から推定されたクラスを取得
	static int calc_estimated_class_index(const std::vector<double> &softmax_output);

};
