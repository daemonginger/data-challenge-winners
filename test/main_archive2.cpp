#include<iostream>
#include<iomanip>
#include"easySD.hpp"
#include<unordered_set>
#include<algorithm>
#include<armadillo>
#include<cmath>
#include<cstdio>
#include<cmath>

using namespace std;
using namespace arma;

vector<int> grams = {1,2};

void save_sparse_matrix(const string& path,const sp_mat& X)
{
	fstream f(path,fstream::out);
	int n = X.n_rows,p = X.n_cols,cmp = 0;
	vector<vector<pair<int,int>>> M(n); 
	for(auto it=X.begin();it!=X.end();it++)
	{
		++cmp;
		M[it.row()].push_back({it.col(),(int)(*it)});
	}
	f << "%%MatrixMarket matrix coordinate integer general\n%\n";
	f << n << " " << p << " " << cmp << endl;
	for(int i=0;i<n;i++)
		for(auto p : M[i])
			f << i+1 << " " << p.first+1 << " " << p.second << endl;
		f.close();
}

void normalize(sp_mat& X)
{
	int p = X.n_cols;
	vector<double> col_norms(p,0.);
	for(sp_mat::const_iterator it=X.begin();it!=X.end();it++)
		col_norms[it.col()] += (*it)*(*it);
	for(int i=0;i<p;i++)
		col_norms[i] = 1/sqrt(col_norms[i]);
	for(sp_mat::iterator it=X.begin();it!=X.end();it++)
		(*it) *= col_norms[it.col()];
}

int main(void)
{
	vector<vector<string> > x_train;
	vector<bool> y_train;
	vector<vector<string> > x_test;
	loaders::load_data("../data/train.csv","../data/test.csv",x_train,y_train,x_test,5,2,1);
	
	// 	fstream f("../data/train_great_stem5_noyou.csv",fstream::out);
	// 	f << "y;message" << endl;
	// 	for(int i=0;i<(int)x_train.size();i++)
	// 	{
	// 		f << y_train[i] << ';';
	// 		for(auto word : x_train[i])
	// 			f << word << " ";
	// 		f << endl;
	// 	}
	
	// 	auto x = x_train;
	// 	x.insert(x.begin(),x_test.begin(),x_test.end())
	sp_mat A = vectorizers::count_vectorize(x_train,grams);
	// 	normalize(A);
	// 	save_sparse_matrix("../../arma_matrix.mtx",A);
	
	int n_samples = A.n_rows,n = (n_samples*9)/10,n_valid = n_samples - n;
	sp_mat X_train = A.head_rows(n), X_valid = A.tail_rows(n_valid);
	vec Y_train(vector<double>(y_train.begin(),y_train.begin() + n)),Y_valid(vector<double>(y_train.begin() + n,y_train.end()));
	
	
	double C = 49.,tol = 1e-5,alpha = 0.00002,correct_rate = 0.1;
	logreg clf = logreg(C,tol,alpha,correct_rate);
	clf.verbose = true;
	clf.fit(X_train,Y_train);
	cout << "Number of samples in train : " << X_train.n_rows << endl << "Number of features (ie size of dictionnary) : " << X_train.n_cols << endl;
	// 	cout << clf.w.head(20) << endl;
	// 	cout << clf.w0 << endl;
	cout << "Score on validation : " << clf.score(X_valid,Y_valid) << endl;
	// 0.812217 with no stemming and min_length = 1
	
	// value = 3242.18 tol = 1e-1
	// value = 3242.1  tol = 1e-2
	// value = 3242.09 tol = 1e-5
	// value = 3242.09 tol = 1e-8
	
	return 0;
}