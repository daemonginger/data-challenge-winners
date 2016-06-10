#ifndef VECTORIZERS_L
#define VECTORIZERS_L

#include<iostream>
#include<string>
#include<vector>
#include<unordered_set>
#include<unordered_map>
#include<map>
#include<set>
#include<algorithm>
#include<armadillo>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/io.hpp>
typedef boost::numeric::ublas::compressed_matrix<double> sparsem;
typedef unsigned char uchar;

namespace vectorizers
{
	std::vector<std::unordered_set<std::string> > hash_vectorize(const std::vector<std::vector<std::string> >&,const std::unordered_set<std::string>&);
	std::vector<std::unordered_map<std::string,int> > count_vectorize(const std::vector<std::vector<std::string> >&,const std::unordered_set<std::string>&);
	std::vector<std::unordered_map<std::string,int> > n_gram_vectorize(const std::vector<std::vector<std::string> >&,const std::vector<int>&);
	arma::sp_mat bin_vectorize(const std::vector<std::vector<std::string> >&);
};

#endif