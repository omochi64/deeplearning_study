#include <string>
#include <fstream>
#include <vector>
#include <iostream>

#include "iris_loader.h"
#include "string_util.h"
#include "data.h"

unsigned int iris_from_string(const std::string &str)
{
	if (str == "Iris-setosa")
	{
		return SETOSA;
	}
	else if (str == "Iris-versicolor")
	{
		return VERSICOLOR;
	}
	else if (str == "Iris-virginica")
	{
		return VIRGINICA;
	}

	return -1;
}


bool load_iris_dataset(std::string fname, std::vector<Instance> &load_data, bool normalize_dim)
{
	load_data.clear();

	std::ifstream ifs;
	ifs.open(fname);
	if (ifs.bad())
	{
		std::cerr << "failed to open: " << fname.c_str() << std::endl;
		return false;
	}

	std::string line;
	std::getline(ifs, line);
	int lineNo = 0;
	while (!ifs.eof() && !ifs.bad())
	{
		lineNo++;
		line = string_rstrip(line);
		if (line == "")
		{
			std::getline(ifs, line);
			continue;
		}

		std::vector<std::string> string_data;
		string_split(line, ',', string_data);

		unsigned int iris = iris_from_string(string_data[string_data.size() - 1]);
		if (iris == (unsigned int)(-1))
		{
			std::cerr << "iris class wrong: " << string_data[string_data.size() - 1].c_str() << " on line " << lineNo << std::endl;
			return false;
		}

		Instance inst;
		inst.cls = iris;
		inst.data.reserve(string_data.size() - 1);

		for (int i = 0; i < string_data.size() - 1; i++)
		{
			inst.data.push_back(atof(string_data[i].c_str()));
		}

		load_data.push_back(inst);

		std::getline(ifs, line);
	}

	ifs.close();

	if (normalize_dim)
	{
		std::vector<double> sum, sq_sum;
		sum.resize(load_data[0].data.size());
		sq_sum.resize(load_data[0].data.size());

		for (size_t i = 0; i < load_data.size(); i++)
		{
			auto &data = load_data[i];

			for (size_t j = 0; j < data.data.size(); j++)
			{
				sum[j] += data.data[j];
				sq_sum[j] += data.data[j] * data.data[j];
			}
		}

		std::vector<double> avg, stdev;
		for (size_t j = 0; j < sum.size(); j++)
		{
			avg.push_back(sum[j] / load_data.size());
			stdev.push_back(std::sqrt(sq_sum[j] / load_data.size() - avg[j] * avg[j]));
		}
		size_t count = load_data.size();
		for (size_t i = 0; i < load_data.size(); i++)
		{
			auto &data = load_data[i];

			for (size_t j = 0; j < data.data.size(); j++)
			{
				data.data[j] = (data.data[j] - avg[j]) / stdev[j];
			}
		}
	}

	return true;
}
