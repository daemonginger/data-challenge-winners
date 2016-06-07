#ifndef BERNOUNB_L
#define BERNOUNB_L
#include<cmath>
#include<vector>

// #include <boost/numeric/ublas/matrix.hpp>
// #include <boost/numeric/ublas/matrix_sparse.hpp>
// #include <boost/numeric/ublas/io.hpp>
// typedef boost::numeric::ublas::compressed_matrix<double> sparsem;

class bernouilli_naive_bayes
{
private:
	// Number of different words in documents
	int n_features;
	// Laplace regularization parameter
	double alpha = 0.;
	// Number of times each word is present in document for classes separately
// 	sparsem occurances;
	// Log-probability of words given a class
// 	sparsem log_prob;
// 	sparsem neg_log_prob;
// 	log-smoothed-Number of documents in classes separately
	int docs[2] = {0};
	// Log-prior of classes
	double log_prior[2] = {0.};
	// Entire vocabulary of the fitted documents
// 	std::unordered_set<std::string> voc;
public:
// 	void fit(const sparsem&,const std::vector<bool>&);
// 	std::vector<bool> predict(const sparsem&);
	bernouilli_naive_bayes(const double&);
};

#endif