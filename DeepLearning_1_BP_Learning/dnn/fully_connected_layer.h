#pragma once

#include "layer.h"
#include "fully_connected_neuron.h"

class ActivationAbst;

//-----HiddenLayer(FC)-----//
class FullConnectedLayer : public NetworkLayer
{
public:
	explicit FullConnectedLayer(unsigned int neuron_count, unsigned int prev_layer_output_count, std::shared_ptr<ActivationAbst> activator, bool add_bias_term = false);
	void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true) override;
	// in_deriv_by_out_x_delta_of_next_layer: list of [d(In_nextLayer)/d(Out_thisLayer) * delta_nextLayer]: size = neuron_count_of_this_layer
	void update_weights_by_bp(const std::vector<std::vector<double> > &in_deriv_out_x_delta_of_next_layer,
		double learning_rate,
		std::vector<std::vector<double> > &o_in_deriv_out_x_delta_of_this_layer);

	size_t get_neuron_count() const override
	{
		return neurons_.size() + (add_bias_term_ ? 1 : 0);
	}
	size_t get_neuron_count_wo_bias() const
	{
		return neurons_.size();
	}

	void print_weights();
	

	mutable double calc_delta_past_sec_, update_weights_past_sec_, calc_bp_loss_past_sec_;

private:
	std::vector<FullConnectedNeuron> neurons_;
	unsigned int                     prev_layer_output_count_;
	bool                             add_bias_term_;
	std::vector<std::vector<double> > latest_input_;

};
