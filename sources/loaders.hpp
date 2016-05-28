#ifndef LOADERS_L
#define LOADERS_L

#include<string>
#include<vector>
#include<fstream>

namespace loaders
{
	const int min_size = 4,train_size = 4415,test_size = 4414;
	void load_train(const std::string&,std::vector<std::vector<std::string> >&,std::vector<bool>&);
	void load_test(const std::string&,std::vector<std::vector<std::string> >&);
};

#endif