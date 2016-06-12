#ifndef LOADERS_L
#define LOADERS_L

#include<string>
#include<vector>
#include<fstream>
#include<sstream>
#include<boost/regex.hpp>

namespace loaders
{
	const int min_size = 4,train_size = 4415,test_size = 4414;
	void load_train(const std::string&,std::vector<std::vector<std::string> >&,std::vector<bool>&);
    void loaders::load_data(const std::string&, 
                            const std::string&, 
                            std::vector<std::vector<std::string> >&, 
                            std::vector<bool>&,
                            std::vector<std::vector<std::string> >&);
    void load_smileys(const std::string&, std::vector<std::pair<std::string, bool>>&);
    void load_corrections(const std::string&, std::vector<std::pair<std::string, std::string>>>&);
    void getRegexps(std::vector<std::pair<std::string, std::string>>&);
	void load_test(const std::string&,std::vector<std::vector<std::string> >&);
};

#endif