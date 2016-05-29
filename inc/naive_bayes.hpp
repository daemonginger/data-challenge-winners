#ifndef NAIVEBAYES_L
#define NAIVEBAYES_L

#include<unordered_map>
#include<vector>
#include<string>

// Implementation inspired by https://www.youtube.com/watch?v=DdYSMwEWbd4
// The idea is that to maximise P(C|X), one should maximise product(P(Xi|C))*P(C).
// P(C) = (number of documents of class C)/(number of documents). Since denominator doesn't depend on C, we won't include it in the implementation.
// We say that (and we are allowed to doubt it) P(Xi|C) = (number of occurances of Xi in all documents of class C + alpha)
//                                                       /(total number of words in all documents of class C + alpha*(size of vocabulary of class C))

class naive_bayes
{
private:
	// Laplace regularization parameter
	double alpha = 0.;
	// Number of occurances of each word in classes separately
	std::unordered_map<std::string,int> occurances[2];
	// Total number of words in all documents in classes separately
	int total[2] = {0};
	// Size of vocabulary (number of distinct words) in classes separately
	int voca[2] = {0};
	// Number of documents in classes separately
	int docs[2] = {0};
	
public:
	naive_bayes(const double&);
	void fit(const std::vector<std::unordered_map<std::string,int> >&,const std::vector<bool>&);
	std::vector<bool> predict(const std::vector<std::unordered_map<std::string,int> >&);
	std::vector<bool> predict2(const std::vector<std::unordered_map<std::string,int> >&);
};

#endif