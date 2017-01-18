
// call cuda kernel to perform update weights
//  delta_mat: matrix of delta [row,column] = [neuron, sample]:   must be a single array
//  in_mat   : matrix of input [row,column] = [data_dim, sample]: must be a single array
//  weights  : matrix of weight[row,column] = [neuron, data_dim]: must be a single array
//  o_weights_diff: output matrix of weight[row, column] = [neuron, data_dim] : must be a single array
void cuda_fc_calc_weights_diff_by_bp(int data_count, int neuron_count, int in_data_dim, float learning_rate, float *delta_mat, float *in_data_mat, float *weights, float *o_weights_diff);
