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

using namespace arma;

namespace vectorizers
{
	sp_mat count_vectorize(const std::vector<std::vector<std::string> >&);
	sp_mat bin_vectorize(const std::vector<std::vector<std::string> >&);
};

#endif