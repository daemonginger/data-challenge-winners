#include"bernouilli_naive_bayes.hpp"

using namespace boost::numeric::ublas;

// Inspired by this http://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html
// And by the implementation of scikit-learn.

bernouilli_naive_bayes::bernouilli_naive_bayes(const double& _alpha):alpha(_alpha) {}

void bernouilli_naive_bayes::fit(const sparsem& text,const std::vector<bool>& labels)
{
	int n = text.size1();
	n_features = text.size2();
	sparsem Y(2,n);
	for(int j=0;j<2;j++)
		for(int i=0;i<n;i++)
				Y.push_back(j,i,j^labels[i]);
	
	
	// Please, don't try this at home or at school.
// 	occurances = Y*text;

	occurances = prod(Y,text);
	for(int j=0;j<2;j++)
		for(int i=0;i<n;i++)
			docs[j] += Y(j,i);
		
	for(int j=0;j<2;j++)
		log_prior[j] = log(docs[j]) - log(docs[0] + docs[1]);
	double inv_docs[2];
	for(int j=0;j<2;j++)
	{
		inv_docs[j] = 1./(docs[j] + 2*alpha);
		docs[j] = log(docs[j] + 2*alpha);
	}
	
	log_prob = sparsem(2,n_features);
	neg_log_prob = sparsem(2,n_features);
	for(int j=0;j<2;j++)
		for(int i=0;i<n_features;i++)
		{
			log_prob.push_back(j,i,log(occurances(j,i) + alpha) - docs[j]);
			neg_log_prob.push_back(j,i,log(1. - (occurances(j,i) + alpha)*inv_docs[j]));
		}
}

std::vector<bool> bernouilli_naive_bayes::predict(const sparsem& text)
{
	int n = text.size1();
	std::vector<bool> ans(n);
	sparsem n_text = sparsem(n,n_features,1.) - text;
// 	for(int i=0;i<n;i++)
// 		for(int j=0;j<n_features;j++)
// 			n_test.push_back(i,j,1. - text(i,j));
	sparsem post_log_prob = prod(neg_log_prob,trans(n_text));
	
	for(int i=0;i<n;i++)
		ans[i] = (post_log_prob(i,0) > post_log_prob(i,1));
	
	return ans;
}