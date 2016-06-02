#include"vectorizers.hpp"

using namespace boost::numeric::ublas;

std::vector<std::unordered_set<std::string> > vectorizers::hash_vectorize(const std::vector<std::vector<std::string> >& text,const std::unordered_set<std::string>& stopwords = std::unordered_set<std::string>())
{
	std::vector<std::unordered_set<std::string> > hashed_text(text.size());
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			if(stopwords.find(word) == stopwords.end())
				hashed_text[i].insert(word);
		}
	}
	return hashed_text;
}

std::vector<std::unordered_map<std::string,int> > vectorizers::count_vectorize(const std::vector<std::vector<std::string> >& text,const std::unordered_set<std::string>& stopwords = std::unordered_set<std::string>())
{
	std::vector<std::unordered_map<std::string,int> > counted_text(text.size());
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			if(stopwords.find(word) == stopwords.end())
				++counted_text[i][word];
		}
	}
	return counted_text;
}

// Results are better without n-grams for now. We may find some better way to use it.

std::vector<std::unordered_map<std::string,int> > vectorizers::n_gram_vectorize(const std::vector<std::vector<std::string> >& text,const std::vector<int>& grams)
{
	std::vector<std::unordered_map<std::string,int> > counted_text(text.size());
	
	for(int n : grams)
	{
		for(int i=0;i<(int)text.size();i++)
		{
			auto sample = text[i];
			for(int j=0;j+n-1<(int)sample.size();j++)
			{
				std::string big_word = "";
				for(int k=j;k<j+n;k++)
				{
					std::string word = sample[k];
					transform(word.begin(),word.end(),word.begin(),::tolower);
					big_word += word;
					if(k != j+n-1)
						big_word += ' ';
				}
				++counted_text[i][big_word];
			}
		}
	}
	return counted_text;
}

sparsem vectorizers::bin_vectorize(const std::vector<std::vector<std::string> >& text)
{
	std::map<std::string,int> voca;
	for(auto sample : text)
		for(auto word : sample)
			voca[word] = 1;
		
	int cmp = 0;
	for(auto& p : voca)
		p.second = cmp++;
	sparsem ans(text.size(), cmp);
	
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		sort(sample.begin(),sample.end());
		for(auto word : sample)
		{
			if(!ans(i,voca[word]))
				ans.push_back(i,voca[word],1.);
		}
	}
	
	return ans;
}