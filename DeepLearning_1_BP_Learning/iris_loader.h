#pragma once


enum IRIS_CLASSES
{
	SETOSA, VERSICOLOR, VIRGINICA, COUNT_LABEL
};


struct Instance;

unsigned int iris_from_string(const std::string &str);
bool load_iris_dataset(std::string fname, std::vector<Instance> &load_data, bool normalize_dim = false);
