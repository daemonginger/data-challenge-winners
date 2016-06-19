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

void dump(const string& path,const vector<vector<string>>& X)
{
	fstream f(path,fstream::out);
	for(auto sample : X)
	{
		for(auto word : sample)
			f << word << " ";
		f << endl;
	}
	f.close();
}

int main(void)
{
	vector<vector<string> > x_train;
	vector<bool> y_train;
	vector<vector<string> > x_test;
	loaders::load_data("../data/train.csv","../data/test.csv",x_train,y_train,x_test,5,2,1);
	
	// 	dump("test_parser.txt",x_test);
	
	// 	fstream f("../data/train_great_stem5_noyou.csv",fstream::out);
	// 	f << "y;message" << endl;
	// 	for(int i=0;i<(int)x_train.size();i++)
	// 	{
	// 		f << y_train[i] << ';';
	// 		for(auto word : x_train[i])
	// 			f << word << " ";
	// 		f << endl;
	// 	}
	// 	f.close();
	
	auto x = x_train;
	int n_train = x_train.size(),n_test = x_test.size();
	x.insert(x.end(),x_test.begin(),x_test.end());
	sp_mat A = vectorizers::count_vectorize(x,grams);
	
	// 	int n_samples = A.n_rows,n = (n_samples*9)/10,n_valid = n_samples - n;
	sp_mat X_train = A.head_rows(n_train), X_test = A.tail_rows(n_test);
	vec Y_train(vector<double>(y_train.begin(),y_train.end()));
	
	
	double C = 51.,tol = 1e-5,alpha = 0.0000001,correct_rate=0.1;
	logreg clf = logreg(C,tol,alpha,correct_rate);
	clf.verbose = true;
	clf.fit(X_train,Y_train);
	cout << "Number of samples in train : " << X_train.n_rows << endl << "Number of features (ie size of dictionnary) : " << X_train.n_cols << endl;
	cout << clf.w.head(10) << endl;
	cout << clf.w0 << endl;
	// 	cout << "Score on validation : " << clf.score(X_valid,Y_valid) << endl;
	uvec Y_pred = clf.predict(X_test);
	fstream f("../subs/sub_4_stem5_youstem_logreg.txt",fstream::out);
	for(int i=0;i<n_test;i++)
		f << Y_pred(i) << endl;
	f.close();
	return 0;
}