#ifndef VECTORIZERS_L
#define VECTORIZERS_L

#include<string>
#include<vector>
#include<unordered_set>
#include<unordered_map>
#include<algorithm>

namespace vectorizers
{
	std::vector<std::unordered_set<std::string> > hash_vectorize(const std::vector<std::vector<std::string> >&);
	std::vector<std::unordered_map<std::string,int> > count_vectorize(const std::vector<std::vector<std::string> >&);
};

#endif