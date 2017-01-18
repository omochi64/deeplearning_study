#pragma once

#include <vector>
#include <memory>


//-----hidden&output layer-----//
class NetworkLayer
{
public:
	NetworkLayer()
		:next_(nullptr)
		, prev_(nullptr)
	{

	}

	virtual void activation(const std::vector<std::vector<float> > &in, std::vector<std::vector<float> > &o_out, bool use_result_to_bp = true) = 0;
	virtual size_t get_neuron_count() const = 0;

	void activation_chain(const std::vector<std::vector<float> > &in, std::vector<std::vector<float> > &o_lastOut)
	{
		std::vector<std::vector<float> > my_result;

		activation(in, my_result);
		if (next_)
		{
			next_->activation_chain(my_result, o_lastOut);
		}
		else
		{
			o_lastOut = my_result;
		}
	}

	void set_next(std::shared_ptr<NetworkLayer> next)
	{
		if (next_)
		{
			next_->prev_ = nullptr;
		}
		next_ = next;
		if (next_)
		{
			next_->prev_ = this;
		}

	}

	void update_bp_from_output_layer(const std::vector<std::vector<float> > &output_layer_loss, float learning_rate)
	{
		// output layer は重み更新しないので、prevにパス
		if (prev_)
		{
			prev_->update_weights_by_bp_chain(output_layer_loss, learning_rate);
		}
	}

	void update_weights_by_bp_chain(const std::vector<std::vector<float> > &next_layer_losses, float learning_rate)
	{
		std::vector<std::vector<float> > cur_loss = next_layer_losses;
		NetworkLayer *cur = this;

		while (cur != nullptr)
		{
			std::vector<std::vector<float> > this_loss;
			cur->update_weights_by_bp(cur_loss, learning_rate, this_loss);

			cur = cur->prev_;
			cur_loss = this_loss;
		}
	}

protected:
	virtual void update_weights_by_bp(const std::vector<std::vector<float> > &next_layer_losses, float learning_rate, std::vector<std::vector<float> > &this_layer_loss_to_bp) = 0;

protected:
	std::shared_ptr<NetworkLayer> next_;
	NetworkLayer                  *prev_;
};
