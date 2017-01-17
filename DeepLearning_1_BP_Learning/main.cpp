#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cassert>
#include <memory>
#include <ctime>
#include <thread>

#include "data.h"
#include "pgm.h"
#include "iris_loader.h"
#include "mnist_loader.h"

#define SIMPLE_TIMER_ENABLE

#ifdef SIMPLE_TIMER_ENABLE

class SimpleTimer
{
public:
	SimpleTimer()
		: begin_(clock())
	{
	}

	double past_seconds() const
	{
		return (double)(clock() - begin_) / CLOCKS_PER_SEC;
	}

private:
	clock_t begin_;
};

#else

class SimpleTimer
{
public:
	SimpleTimer() {}

	double past_seconds() const {
		return 0;
	}
};

#endif




//-----activation-----//

// activation class
class ActivationAbst
{
public:
	virtual double activation(double in)  const = 0;
	virtual double calc_deriv(double in) const = 0;
};
// sigmoid
class SigmoidActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return 1 / (1 + std::exp(-in));
	}
	virtual double calc_deriv(double in) const override
	{
		auto act = activation(in);
		return act * (1 - act);
	}
};
// ReLU
class ReLUActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return in >= 0 ? in : 0;
	}
	virtual double calc_deriv(double in) const override
	{
		return in > 0 ? 1 : 0;
	}
};
// 恒等写像
class IdentityActivation : public ActivationAbst
{
public:
	virtual double activation(double in) const override
	{
		return in;
	}
	virtual double calc_deriv(double in) const override
	{
		return 1;
	}
};


//-----hidden&output layer-----//
class NetworkLayer
{
public:
	NetworkLayer()
		:next_(nullptr)
		,prev_(nullptr)
	{

	}

	virtual void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true)  = 0;
	virtual size_t get_neuron_count() const = 0;

	void activation_chain(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_lastOut)
	{
		std::vector<std::vector<double> > my_result;

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

	void update_bp_from_output_layer(const std::vector<std::vector<double> > &output_layer_loss, double learning_rate)
	{
		// output layer は重み更新しないので、prevにパス
		if (prev_)
		{
			prev_->update_weights_by_bp_chain(output_layer_loss, learning_rate);
		}
	}

	void update_weights_by_bp_chain(const std::vector<std::vector<double> > &next_layer_losses, double learning_rate)
	{
		std::vector<std::vector<double> > cur_loss = next_layer_losses;
		NetworkLayer *cur = this;

		while (cur != nullptr)
		{
			std::vector<std::vector<double> > this_loss;
			cur->update_weights_by_bp(cur_loss, learning_rate, this_loss);

			cur = cur->prev_;
			cur_loss = this_loss;
		}
	}

protected:
	virtual void update_weights_by_bp(const std::vector<std::vector<double> > &next_layer_losses, double learning_rate, std::vector<std::vector<double> > &this_layer_loss_to_bp) = 0;

protected:
	std::shared_ptr<NetworkLayer> next_;
	NetworkLayer                  *prev_;
};
// ニューロン1個
class FullConnectedNeuron
{
public:
	explicit FullConnectedNeuron(unsigned int input_count, std::shared_ptr<ActivationAbst> activator)
		: activator_(activator)
		
	{
		weights_.resize(input_count);
		for (size_t i = 0; i < input_count; i++) weights_[i] = 0;

		// random init
		for (size_t i = 0; i < input_count; i++)
		{
			weights_[i] = ((rand() % 1001) / 1000.0f - 0.5) * 2;
		}
	}
	void activation(const std::vector<std::vector<double> > &in, std::vector<double> &o_out, bool save_activ_deriv = true)
	{
		assert(in[0].size() == weights_.size());
		o_out.resize(in.size());

		if (save_activ_deriv && latest_activ_deriv_.size() != in.size()) latest_activ_deriv_.resize(in.size());

		for (size_t data_i = 0; data_i < in.size(); data_i++)
		{
			double sum = 0;
			for (size_t i = 0; i < weights_.size(); i++)
			{
				sum += weights_[i] * in[data_i][i];
			}

			if (save_activ_deriv)
			{
				latest_activ_deriv_[data_i] = activator_->calc_deriv(sum);
			}

			o_out[data_i] = activator_->activation(sum);
		}
		
	}

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
	void update_weights(const std::vector<std::vector<double> > &in_from_prev_layer, const std::vector<double> &delta, double learning_rate)
	{
		assert(in_from_prev_layer[0].size() == weights_.size());

		size_t size = weights_.size();
		double *weights = weights_.data();

		for (size_t i = 0; i < size; i++)
		{
			double error_sum = 0;

			for (size_t data_i = 0; data_i < in_from_prev_layer.size(); data_i++)
			{
				error_sum += in_from_prev_layer[data_i][i] * delta[data_i];
			}

			weights[i] -= learning_rate * error_sum / in_from_prev_layer.size();
		}
	}

	void print_weights()
	{
		std::cout << "[";
		for (auto weight : weights_)
		{
			printf("%.2f ", weight);
		}
		std::cout << "]" << std::endl;
	}

private:
	std::vector<double> weights_;
	std::vector<double>  latest_activ_deriv_;
	std::shared_ptr<ActivationAbst> activator_;
};

//-----HiddenLayer(FC)-----//
class FullConnectedLayer : public NetworkLayer
{
public:
	explicit FullConnectedLayer(unsigned int neuron_count, unsigned int prev_layer_output_count, std::shared_ptr<ActivationAbst> activator, bool add_bias_term = false)
		: NetworkLayer()
		, add_bias_term_(add_bias_term)
		, prev_layer_output_count_(prev_layer_output_count)
	{
		neurons_.clear();
		neurons_.reserve(neuron_count);

		for (size_t i = 0; i < neuron_count; i++)
		{
			neurons_.emplace_back(prev_layer_output_count, activator);
		}
	}
	void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true) override
	{
		if (use_result_to_bp)
		{
			latest_input_ = in;
		}

		size_t data_count = in.size();
		o_out.clear();
		o_out.resize(data_count);

		size_t neuron_count = neurons_.size();
		auto *neurons_array = neurons_.data();

		std::vector<double> *out_array = o_out.data();

		for (size_t data_i = 0; data_i < data_count; data_i++)
		{
			o_out[data_i].resize(neuron_count);
			if (add_bias_term_)
			{
				// add_bias_term が true の場合、出力にBiasとなる「1」だけを出力するニューロンを追加する
				// *入力にbiasを入れるのではなく、この層がbiasを出力する
				o_out[data_i].push_back(1);
			}
		}

//#pragma omp parallel
		for (size_t i = 0; i < neuron_count; i++)
		{
			auto &neuron = neurons_array[i];

			std::vector<double> activated_values;
			neuron.activation(in, activated_values, use_result_to_bp);

			for (size_t data_i = 0; data_i < data_count; data_i++)
			{
				o_out[data_i][i] = activated_values[data_i];
			}
		}
	}

	// in_deriv_by_out_x_delta_of_next_layer: list of [d(In_nextLayer)/d(Out_thisLayer) * delta_nextLayer]: size = neuron_count_of_this_layer
	void update_weights_by_bp(const std::vector<std::vector<double> > &in_deriv_out_x_delta_of_next_layer,
		double learning_rate,
		std::vector<std::vector<double> > &o_in_deriv_out_x_delta_of_this_layer)
	{
		assert(in_deriv_out_x_delta_of_next_layer[0].size() == get_neuron_count());

		std::vector<std::vector<double> > delta;
		delta.resize(in_deriv_out_x_delta_of_next_layer.size());

		auto *neurons_ptr  = neurons_.data();
		std::vector<double> *delta_ptr    = delta.data();
		auto *in_deriv_ptr = in_deriv_out_x_delta_of_next_layer.data();

		SimpleTimer delta_timer;

		// この層のニューロンごとのdeltaを求める
		// ニューロンのdelta = 活性化関数の微分値(ニューロンへの入力値) * 前段からの計算値
		for (size_t data_i = 0; data_i < delta.size(); data_i++)
		{
			delta[data_i].resize(get_neuron_count_wo_bias());
			for (size_t i = 0; i < get_neuron_count_wo_bias(); i++)
			{
				double delta_i = neurons_ptr[i].get_latest_activ_deriv()[data_i] * in_deriv_ptr[data_i][i];
				delta[data_i][i] = delta_i;
			}
		}

		calc_delta_past_sec_ = delta_timer.past_seconds();

		SimpleTimer calc_bp_loss_timer;

		// 次のBP用に「d この層への入力(activation直前)_per_this_layer_neuron/ d 前の層の出力_per_prev_layer_neuron * delta」を求める
		// 「前段からの計算値」に相当
		// ↑の値は、this layer neuron 全てに対して和をとる
		// ↑の微分は、FullyConnectedの場合はニューロン間の重みになる
		//std::vector<double> next_bp_output;  // サイズは prev_layer_output_count_ になる
		//next_bp_output.resize(prev_layer_output_count_);
		o_in_deriv_out_x_delta_of_this_layer.resize(delta.size());
		//o_in_deriv_out_x_delta_of_this_layer.resize(prev_layer_output_count_);
		auto *next_bp_ptr = o_in_deriv_out_x_delta_of_this_layer.data();

		for (size_t data_i = 0; data_i < delta.size(); data_i++)
		{
			next_bp_ptr[data_i].resize(prev_layer_output_count_);
		}
		
#pragma omp parallel
		for (size_t data_i = 0; data_i < delta.size(); data_i++)
		{
			for (size_t i = 0; i < prev_layer_output_count_; i++)
			{
				// この層の全てのニューロンに対して合計を取る
				double value = 0;
				for (size_t k = 0; k < get_neuron_count_wo_bias(); k++)
				{
					// d この層への入力(activation直前)_per_this_layer_neuron/ d 前の層の出力_per_prev_layer_neuron: 重みに相当
					auto in_deriv_out = neurons_ptr[k].calc_deriv_btw_in_out(i);

					value += in_deriv_out * delta[data_i][k];
				}

				next_bp_ptr[data_i][i] = value;
			}
		}
		calc_bp_loss_past_sec_ = calc_bp_loss_timer.past_seconds();

		SimpleTimer update_timer;

		// 計算したdeltaを用いて更新
		// W_ij_next = W_ij_cur - learning_rate * delta_j * Out_i
		size_t neurons_count = get_neuron_count_wo_bias();
#pragma omp parallel
		for (size_t j = 0; j < neurons_count; j++)
		{
			auto &neuron = neurons_ptr[j];
			std::vector<double> delta_for_j;
			for (size_t data_i = 0; data_i < delta.size(); data_i++) delta_for_j.push_back(delta[data_i][j]);
			neuron.update_weights(latest_input_, delta_for_j, learning_rate);
		}

		update_weights_past_sec_ = update_timer.past_seconds();
	}

	size_t get_neuron_count() const override
	{
		return neurons_.size() + (add_bias_term_ ? 1 : 0);
	}
	inline size_t get_neuron_count_wo_bias() const
	{
		return neurons_.size();
	}

	void print_weights()
	{
		for (auto &neuron : neurons_)
		{
			neuron.print_weights();
		}
	}

	mutable double calc_delta_past_sec_, update_weights_past_sec_, calc_bp_loss_past_sec_;

private:
	std::vector<FullConnectedNeuron> neurons_;
	unsigned int                     prev_layer_output_count_;
	bool                             add_bias_term_;
	std::vector<std::vector<double> > latest_input_;

};

//-----output-----//
class SoftmaxLayer : public NetworkLayer
{
public:
	void activation(const std::vector<std::vector<double> > &in, std::vector<std::vector<double> > &o_out, bool use_result_to_bp = true) override
	{
		o_out.clear();
		int dim = in[0].size();
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


	size_t get_neuron_count() const override
	{
		return 1;
	}

	static int calc_estimated_class_index(const std::vector<double> &softmax_output)
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
	void update_weights_by_bp(const std::vector<std::vector<double> > &, double, std::vector<std::vector<double> > &)
	{
		assert(false);
	}

	// ロス計算
	double calc_loss(const std::vector<double> &result, size_t gt_index)
	{
		return -std::log(result[gt_index]); // gt_index以外の項は0になる

	}

	// 微分計算
	void calc_delta(const std::vector<double> &result, const size_t gt_index, std::vector<double> &o_delta)
	{
		assert(gt_index >= 0 && gt_index < result.size());

		o_delta.resize(result.size());

		for (size_t i = 0; i < result.size(); i++)
		{
			o_delta[i] = result[i] - (i == gt_index ? 1 : 0);
		}
	}
};

int main(int argc, char *argv[])
{
	std::string fname = "iris_data_dummy.csv";

	//std::vector<Instance> iris_dataset;
	//load_iris_dataset(fname, iris_dataset, true);

	std::vector<Instance> dataset;
	std::vector<Instance> testset;

	bool is_mnist = false;

	if (is_mnist)
	{
		load_mnist_dataset("train-images.idx3-ubyte", "train-labels.idx1-ubyte", dataset);
		load_mnist_dataset("t10k-images.idx3-ubyte", "t10k-labels.idx1-ubyte", testset);
	}
	else
	{
		load_iris_dataset(fname, dataset, true);
		load_iris_dataset(fname, testset, true);
	}
	

	// input - 5(+1) - 4(+1) - labels - softmax
	auto activ = std::make_shared<IdentityActivation>();

	std::shared_ptr<FullConnectedLayer> first_layer;
	first_layer = std::make_shared<FullConnectedLayer>(10, (unsigned int)(dataset[0].data.size()+1), activ, false);
	//auto second = std::make_shared<FullConnectedLayer>(150, first_layer->get_neuron_count(), activ, false);
	//first_layer->set_next(second);
	//auto third = std::make_shared<FullConnectedLayer>(10, first_layer->get_neuron_count(), activ, false);
	//first_layer->set_next(third);
	auto softmax = std::make_shared<SoftmaxLayer>();
	first_layer->set_next(softmax);

	int batch_size = 100;
	double bias_init = 0.1;

	// update by bp
	for (size_t bp = 0; bp < 1000; bp++)
	{
		std::cout << bp << "th update" << std::endl;
		
		unsigned int correct_count = 0;

		double time_for_forward = 0;
		double time_for_bp = 0;

		double time_for_calc_delta = 0;
		double time_for_calc_loss = 0;
		double time_for_update = 0;

		for (int i = 0; i < dataset.size();)
		{
			// feed forward
			std::vector<std::vector<double> > input_array; input_array.reserve(batch_size);
			std::vector<unsigned int> label_array;
			for (int batch_i = 0; batch_i < batch_size; batch_i++)
			{
				if (batch_i + i >= dataset.size()) continue;
				auto input_data = dataset[batch_i + i].data;
				input_data.push_back(bias_init);
				input_array.push_back(input_data);
				label_array.push_back(dataset[batch_i+i].cls);
			}
			
			std::vector<std::vector<double> > output_array;

			SimpleTimer timer_forward;

			first_layer->activation_chain(input_array, output_array);

			// get the predicted class
			std::vector<std::vector<double> > loss_delta(output_array.size());
			for (size_t data_i = 0; data_i < output_array.size(); data_i++)
			{
				int max_idx = SoftmaxLayer::calc_estimated_class_index(output_array[data_i]);

				//std::cout << i << ": " << iris.cls << " est:" << max_idx << " ," << (iris.cls == max_idx ? "correct" : "wrong") << std::endl;
				if (label_array[data_i] == max_idx) correct_count++;

				softmax->calc_delta(output_array[data_i], label_array[data_i], loss_delta[data_i]);
			}
			time_for_forward += timer_forward.past_seconds();

			SimpleTimer timer_bp;

			// backprop
			softmax->update_bp_from_output_layer(loss_delta, 0.01);

			time_for_bp += timer_bp.past_seconds();

			time_for_calc_delta += /*third->calc_delta_past_sec_*/ + first_layer->calc_delta_past_sec_;
			time_for_calc_loss += /*third->calc_bp_loss_past_sec_*/ + first_layer->calc_bp_loss_past_sec_;
			time_for_update += /*third->update_weights_past_sec_*/ + first_layer->update_weights_past_sec_;

			int next_i = i + batch_size;
			if      (dataset.size() * 3 / 4 >= i && dataset.size() * 3 / 4 < next_i) std::cout << "75% of epoch finished" << std::endl;
			else if (dataset.size() * 2 / 4 >= i && dataset.size() * 2 / 4 < next_i) std::cout << "50% of epoch finished" << std::endl;
			else if (dataset.size() * 1 / 4 >= i && dataset.size() * 1 / 4 < next_i) std::cout << "25% of epoch finished" << std::endl;

			i = next_i;
		}

		std::cout << "Time for forward, bp = " << time_for_forward << "," << time_for_bp << std::endl;
		std::cout << "Time for calc delta, loss, update = " << time_for_calc_delta << "," << time_for_calc_loss << "," << time_for_update << std::endl;

		std::cout << "correct rate (training): " << correct_count << "/" << dataset.size() << " = "  << (100.0 * correct_count / dataset.size()) << "%" << std::endl;

		if (true)
		{
			double loss_total = 0;
			int correct_total = 0;
			std::cout << "calculating test loss..." << std::endl;
			for (size_t i = 0; i < testset.size(); i++)
			{
				auto &data = testset[i];
				std::vector<double> input(data.data);
				input.push_back(bias_init);  // bias term
				std::vector<std::vector<double> > input_array(1); input_array[0] = input;
				std::vector<std::vector<double> > output_array;

				first_layer->activation_chain(input_array, output_array);
				int max_idx = SoftmaxLayer::calc_estimated_class_index(output_array[0]);
				if (max_idx == data.cls) correct_total++;

				loss_total += softmax->calc_loss(output_array[0], data.cls);
			}
			std::cout << "test loss (total): " << loss_total << std::endl;
			std::cout << "test correct rate: " << correct_total << "/" << testset.size() << " = " << (100.0f * correct_total / testset.size()) << "%" << std::endl;
		}
	}

	

	return 0;
}
