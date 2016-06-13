#include"vectorizers.hpp"

sp_mat vectorizers::bin_vectorize(const std::vector<std::vector<std::string> >& text,const vector<int>& grams)
{
	unsigned int min_size = 1;
	std::map<std::string,int> voca;
	int tot_size = 0;
	for(auto sample : text)
	{
		std::set<std::string> faitchier;
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				faitchier.insert(word);
			}
		}
		tot_size += faitchier.size();
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				voca[word] = 1;
			}
		}
	}
	umat A(2,tot_size);
	int cmp = 0;
	for(auto& p : voca)
		p.second = cmp++;
	
	cmp = 0;
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		std::set<std::string> faitchier;
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				faitchier.insert(word);
			}
		}
		for(auto word : faitchier)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				A(0,cmp) = i;
				A(1,cmp++) = voca[word];
			}
		}
	}
	return sp_mat(A,ones<vec>(tot_size));
}

sp_mat vectorizers::count_vectorize(const std::vector<std::vector<std::string> >& text,const vector<int>& grams)
{
	unsigned int min_size = 1;
	std::map<std::string,int> voca;
	int tot_size = 0;
	for(auto sample : text)
	{
		std::set<std::string> faitchier;
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				faitchier.insert(word);
			}
		}
		tot_size += faitchier.size();
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				voca[word] = 1;
			}
		}
	}
	umat A(2,tot_size);
	int cmp = 0;
	for(auto& p : voca)
		p.second = cmp++;
	
	std::vector<double> vals(tot_size);
	
	cmp = 0;
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		std::map<std::string,int> faitchier;
		for(auto word : sample)
		{
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				++faitchier[word];
			}
		}
		for(auto pa : faitchier)
		{
			auto word = pa.first;
			if(word.size() > min_size)
			{
				transform(word.begin(),word.end(),word.begin(),::tolower);
				A(0,cmp) = i;
				A(1,cmp) = voca[word];
				vals[cmp++] = pa.second;
			}
		}
	}
	return sp_mat(A,vec(vals));
}