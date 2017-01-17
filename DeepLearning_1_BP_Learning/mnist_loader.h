#pragma once

#include <vector>
#include <string>

struct Instance;

bool load_mnist_dataset(std::string fname, const std::string &label_fname, std::vector<Instance> &load_data, bool normalize_dim = false);

