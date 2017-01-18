
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <math.h>

// perform: weights = weights - learning_rate * delta_weights
//          delta_weights = (1/data_count) * delta_mat * in_mat^T
//  delta_mat: matrix of delta [row,column] = [neuron, sample]
//  in_mat   : matrix of input [row,column] = [data_dim, sample]
//  weights  : matrix of weight[row,column] = [neuron, data_dim]
__global__ void kernel_fc_calc_weights_diff_by_bp(int data_count, int neuron_count, int in_data_dim, float learning_rate, float *delta_mat, float *in_mat, float *weights, float *weights_diff)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= neuron_count * in_data_dim) return;

	int neuron_idx = index % neuron_count;
	int target_dim = index / neuron_count;

	float *delta_begin = &delta_mat[neuron_idx*data_count];
	float *in_begin    = &in_mat[target_dim*data_count];

	float sum = 0;
	for (int data_i = 0; data_i < data_count; data_i++)
	{
		float delta = delta_begin[data_i];
		float in = in_begin[data_i];

		sum += delta * in;
	}

	weights_diff[neuron_idx * in_data_dim + target_dim] = learning_rate * sum / (data_count);
}

// call cuda kernel
//  delta_mat: matrix of delta [row,column] = [neuron, sample]:   must be a single array
//  in_mat   : matrix of input [row,column] = [data_dim, sample]: must be a single array
//  weights  : matrix of weight[row,column] = [neuron, data_dim]: must be a single array
//  o_weights_diff: output matrix of weight[row, column] = [neuron, data_dim] : must be a single array
void cuda_fc_calc_weights_diff_by_bp(int data_count, int neuron_count, int in_data_dim, float learning_rate, float *delta_mat, float *in_data_mat, float *weights, float *o_weights_diff)
{
	float *k_delta_mat;
	float *k_in_data_mat;
	float *k_weights, *k_weights_diff;

//#define CHECK_RESULT


	cudaMalloc(&k_delta_mat, neuron_count * data_count * sizeof(float));
	cudaMalloc(&k_in_data_mat, data_count   * in_data_dim * sizeof(float));
	cudaMalloc(&k_weights, neuron_count * in_data_dim * sizeof(float));
	cudaMalloc(&k_weights_diff, neuron_count * in_data_dim * sizeof(float));

	cudaMemcpy(k_delta_mat, delta_mat, neuron_count * data_count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(k_in_data_mat, in_data_mat, data_count   * in_data_dim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(k_weights, weights, neuron_count * in_data_dim * sizeof(float), cudaMemcpyHostToDevice);

	int thread_count = neuron_count * in_data_dim;

	// Perform kernel (256 threads in each block)
	kernel_fc_calc_weights_diff_by_bp << <(thread_count + 255) / 256, 256 >> > (data_count, neuron_count, in_data_dim, learning_rate, k_delta_mat, k_in_data_mat, k_weights, k_weights_diff);

	cudaMemcpy(o_weights_diff, k_weights_diff, neuron_count * in_data_dim * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef CHECK_RESULT
	// validate
	// weights[0] (=weights[neuron=0][dim=0]) == prev_weights[0] - learning_rate / data_count * sum(delta[neuron=0] * in[dim=0]) for sample
	for (size_t neuron = 0; neuron < neuron_count; neuron++)
	{
		for (size_t dim = 0; dim < in_data_dim; dim++)
		{
			float sum = 0;
			float cur_v = o_weights_diff[neuron * in_data_dim + dim] / 10000;
			for (size_t i = 0; i < data_count; i++)
			{
				sum += delta_mat[neuron * data_count + i] * in_data_mat[dim * data_count + i];
			}
			float desired = learning_rate * sum / data_count;

			// 学習が進むとlossの値が小さくなって重みがGPUでは更新できなくなる（精度問題）
			if (fabs(desired - cur_v) > 0.000001)
				printf("avg=%f, desired_weight_diff=%f, new_weight_diff=%f\n", sum / data_count, desired, cur_v);
		}
	}
#endif


}
