#include"vectorizers.hpp"

sp_mat vectorizers::bin_vectorize(std::vector<std::vector<std::string> > text,const std::vector<int>& grams)
{
	std::map<std::string,int> voca;
	int tot_size = 0;
	
	std::vector<std::vector<std::string> > text_aux(text.size());
	for(unsigned int n : grams)
	{
		for(int k=0;k<(int)text.size();k++)
		{
			auto sample = text[k];
			for(int i=0;i<(int)sample.size()-(int)n+1;i++)
			{
				std::string word;
				for(unsigned int j=i;(int)j<i+(int)n;j++)
				{
					word += sample[j];
					if(j != i+n-1)
						word += " ";
				}
				transform(word.begin(),word.end(),word.begin(),::tolower);
				text_aux[k].push_back(word);
			}
		}
	}
	text = text_aux;
	
	for(auto sample : text)
	{
		std::set<std::string> sample_voc;
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			sample_voc.insert(word);
		}
		tot_size += sample_voc.size();
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			voca[word] = 1;
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
		std::set<std::string> sample_voc;
		for(auto word : sample)
		{	
			transform(word.begin(),word.end(),word.begin(),::tolower);
			sample_voc.insert(word);
		}
		for(auto word : sample_voc)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			A(0,cmp) = i;
			A(1,cmp++) = voca[word];
		}
	}
	return sp_mat(A,ones<vec>(tot_size));
}

sp_mat vectorizers::count_vectorize(std::vector<std::vector<std::string> > text,const std::vector<int>& grams)
{
	std::map<std::string,int> voca;
	int tot_size = 0;
	
	std::vector<std::vector<std::string> > text_aux(text.size());
	for(unsigned int n : grams)
	{
		for(int k=0;k<(int)text.size();k++)
		{
			auto sample = text[k];
			for(int i=0;i<(int)sample.size()-(int)n+1;i++)
			{
				std::string word;
				for(unsigned int j=i;(int)j<i+(int)n;j++)
				{
					word += sample[j];
					if(j != i+n-1)
						word += " ";
				}
				transform(word.begin(),word.end(),word.begin(),::tolower);
				text_aux[k].push_back(word);
			}
		}
	}
	text = text_aux;
	
	for(auto sample : text)
	{
		std::set<std::string> sample_voc;
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			sample_voc.insert(word);
		}
		tot_size += sample_voc.size();
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			voca[word] = 1;
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
		std::map<std::string,int> sample_voc;
		for(auto word : sample)
		{
			transform(word.begin(),word.end(),word.begin(),::tolower);
			++sample_voc[word];
		}
		for(auto pa : sample_voc)
		{
			auto word = pa.first;
			transform(word.begin(),word.end(),word.begin(),::tolower);
			A(0,cmp) = i;
			A(1,cmp) = voca[word];
			vals[cmp++] = pa.second;
		}
	}
	return sp_mat(A,vec(vals));
}

sp_mat vectorizers::tfidf_vectorize(std::vector<std::vector<std::string>> text,const std::vector<int>& grams){
    sp_mat A = vectorizers::count_vectorize(text, grams);
    
    sp_mat idfMatrix(A.n_cols, A.n_cols);
    for(int i=0;i<(int)A.n_cols;++i){
        sp_mat col = A.col(i);   // Number of documents in which term i appears.
        double idf = log(A.n_rows / (double)(1 + col.n_nonzero));
        idfMatrix(i,i) = idf;
    }
    
    return A * idfMatrix;
}
