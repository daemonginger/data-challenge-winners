#include"naive_bayes.hpp"
#include<iostream>
#include<cmath>

naive_bayes::naive_bayes(const double& _alpha):alpha(_alpha) {}

// Here we simply precalculate the values necessary for the prediction.

void naive_bayes::fit(const std::vector<std::unordered_map<std::string,int> >& text,const std::vector<bool>& labels)
{
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		++docs[labels[i]];
		for(auto p : sample)
		{
			voc.insert(p.first);
			occurances[labels[i]][p.first] += p.second;
			total[labels[i]] += p.second;
		}
	}
	
	for(int i=0;i<2;i++)
		voca[i] = occurances[i].size();
}

// A variation

void naive_bayes::fit2(const std::vector<std::unordered_map<std::string,int> >& text,const std::vector<bool>& labels)
{
	for(int i=0;i<(int)text.size();i++)
	{
		auto sample = text[i];
		++docs[labels[i]];
		for(auto p : sample)
		{
			voc.insert(p.first);
			++occurances[labels[i]][p.first];
			total[labels[i]] += p.second;
		}
	}
	
	for(int i=0;i<2;i++)
		voca[i] = occurances[i].size();
}

// We simply apply the formulas of the model and check which class has the highest probability.

std::vector<bool> naive_bayes::predict(const std::vector<std::unordered_map<std::string,int> >& text)
{
	std::vector<bool> labels(text.size());
	
	for(int i=0;i<(int)text.size();i++)
	{
		double score[2] = {1.};
		for(int k=0;k<2;k++)
		{
			score[k] = docs[k];
			for(auto& p : text[i])
			{
				std::string word = p.first;
				score[k] *= (alpha + occurances[k][word])/(alpha*voca[k] + total[k]);
			}
		}
		labels[i] = (score[1] > score[0]);
	}
	
	return labels;
}

// A variation of predict, to see what performs best.

std::vector<bool> naive_bayes::predict2(const std::vector<std::unordered_map<std::string,int> >& text)
{
	std::vector<bool> labels(text.size());
	
	for(int i=0;i<(int)text.size();i++)
	{
		double score[2] = {1.};
		for(int k=0;k<2;k++)
		{
			score[k] = docs[k];
			for(auto& p : text[i])
			{
				std::string word = p.first;
				
				for(int j=0;j<p.second;j++)
					score[k] *= (alpha + occurances[k][word])/(alpha*voca[k] + total[k]);
			}
		}
		labels[i] = (score[1] > score[0]);
	}
	
	return labels;
}

// Another variation

std::vector<bool> naive_bayes::predict3(const std::vector<std::unordered_map<std::string,int> >& text)
{
	std::vector<bool> labels(text.size());
	
	for(int i=0;i<(int)text.size();i++)
	{
		double score[2] = {1.};
		for(int k=0;k<2;k++)
		{
			score[k] = log(docs[k]);
// 			std::cout << score[k] << std::endl;
			
// 			for(auto p : occurances[k])
// 			{
// 				std::string word = p.first;
// 				if(text[i].find(word) == text[i].end())
// 				{
// 					// 					std::cout << "yo" << std::endl;
// 					score[k] += log(1. - (alpha + occurances[k][word])/(2.*alpha + docs[k]));
// 				}
// 			}
			
			for(auto word : voc)
			{
				if(text[i].find(word) != text[i].end())
				{
// 					std::cout << "yo" << std::endl;
					score[k] += log((alpha + occurances[k][word])/(2*alpha + docs[k]));
				}
				else
				{
// 					std::cout << "hey" << std::endl;
					score[k] += log(1 - (alpha + occurances[k][word])/(2*alpha + docs[k]));
				}
			}
			
// 			for(auto& p : text[i])
// 			{
// 				std::string word = p.first;
// 				
// 				for(int j=0;j<p.second;j++)
// 					score[k] *= (alpha + occurances[k][word])/(alpha + docs[k]);
// 			}
		}
// 		std::cout << score[0] << " " << score[1] << std::endl;
		labels[i] = (score[1] > score[0]);
	}
	
	return labels;
}