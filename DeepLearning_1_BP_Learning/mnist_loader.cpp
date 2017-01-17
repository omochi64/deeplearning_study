#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include "data.h"

unsigned int reverse_bit(unsigned int r)
{
	return ((r & 0xff) << 24) | (((r >> 8) & 0xff) << 16) | (((r >> 16) & 0xff) << 8) | (((r >> 24) & 0xff) << 0);
}

unsigned int read_reverse(FILE *fp)
{
	unsigned int r;
	fread((void *)&r, sizeof(unsigned int), 1, fp);
	return reverse_bit(r);
}

unsigned int read_reverse(std::ifstream &ifs)
{
	unsigned int r;
	ifs.read((char *)&r, 4);
	return reverse_bit(r);
}


bool load_mnist_dataset(std::string fname, const std::string &label_fname, std::vector<Instance> &load_data, bool normalize_dim)
{
	std::ifstream ifs_label;
	FILE *f;
	fopen_s(&f, fname.c_str(), "rb");
	//ifs.open(fname);
	if (f == nullptr)
	{
		std::cerr << "failed to open: " << fname.c_str() << std::endl;
		return false;
	}

	ifs_label.open(label_fname);
	if (ifs_label.bad())
	{
		std::cerr << "failed to open: " << label_fname.c_str() << std::endl;
		fclose(f);
		return false;
	}

	// read_images
	/*
	0000     32 bit integer  0x00000803(2051) magic number
	0004     32 bit integer  60000            number of images
	0008     32 bit integer  28               number of rows
	0012     32 bit integer  28               number of columns
	0016     unsigned byte   ??               pixel
	0017     unsigned byte   ??               pixel
	........
	xxxx     unsigned byte   ??               pixel
	*/

	int data_count = 0;
	int rows, cols;

	unsigned int read_int = 0;

	read_int = read_reverse(f);
	if (read_int != 2051) { fclose(f); std::cerr << "image file's magic number is not 2051" << std::endl; return false; }

	read_int = read_reverse(ifs_label);
	if (read_int != 2049) { fclose(f); std::cerr << "label file's magic number is not 2049" << std::endl; return false; }

	read_int = read_reverse(f);
	data_count = read_int;

	if (data_count != read_reverse(ifs_label))
	{
		fclose(f);
		std::cerr << "count of image and label file are different" << std::endl; return false;
	}

	read_int = read_reverse(f);
	rows = read_int;

	read_int = read_reverse(f);
	cols = read_int;

	std::cerr << "num: " << data_count << " rows: " << rows << " cols: " << cols << std::endl;

	load_data.clear();
	load_data.reserve(data_count);

	unsigned char *bytes = new unsigned char[rows*cols];
	for (size_t i = 0; i < data_count; i++)
	{
		fread((char *)bytes, sizeof(char), rows*cols, f);
		//ifs.read((char *)bytes, rows*cols);

		Instance inst;
		inst.data.reserve(rows*cols);
		for (int j = 0; j < rows*cols; j++)
		{
			//char b;
			//ifs.read(&b, 1);
			//fread(&b, sizeof(char), 1, f);
			inst.data.push_back((unsigned char)bytes[j] / 255.0);
		}

		char label;
		ifs_label.read(&label, 1);
		inst.cls = label;

		//char tmp[1024];
		//sprintf_s(tmp, 1024, "tmp_/%06d_%d.pgm", i, label);
		//PGM::save_pgm_file(tmp, inst.data.data(), 28, 28);


		load_data.push_back(inst);
	}

	delete[] bytes;

	fclose(f);

	return true;

}

