#include"vectorizers.hpp"

std::vector<std::unordered_set<std::string> > vectorizers::hash_vectorize(const std::vector<std::vector<std::string> >& text)
{
	std::vector<std::unordered_set<std::string> > hashed_text(text.size());
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			hashed_text[i].insert(word);
		}
	}
	return hashed_text;
}

std::vector<std::unordered_map<std::string,int> > vectorizers::count_vectorize(const std::vector<std::vector<std::string> >& text)
{
	std::vector<std::unordered_map<std::string,int> > counted_text(text.size());
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			++counted_text[i][word];
		}
	}
	return counted_text;
}