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

vector<int> grams = {1};

// 12/06 : This code shows our first logistic regression without intercept. We managed to do the same as scikit.

double val(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	int n = X.n_rows;
	vec exp_vect = exp(-Y%(X*w + w0));
	return C*sum(log(1. + exp_vect)) + (sum(w%w) + w0*w0)/2.;
}

pair<double,vec> grad(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	int n = X.n_rows;
	// 	cout << "lolol" << endl;
	vec exp_vect = exp(-Y%(X*w + w0));
	// 	cout << "yolo" << endl;
	vec coefs = -Y%exp_vect/(1. + exp_vect);
	// 	cout << "merde" << endl;
	return {C*sum(coefs) + w0,C*(coefs.t()*X).t() + w};
}

int main(void)
{
	cout << "YOLO" << endl;
	vector<vector<string> > x_train;
	vector<bool> y_train;
	vector<vector<string> > x_test;
// 	loaders::load_train("../data/train.csv",x_train,y_train);
// 	loaders::load_test("../data/test.csv",x_test);
	
	loaders::load_data("../data/train_great.csv","../data/test.csv",x_train,y_train,x_test);
	
// 	fstream f("../data/train_great.csv",fstream::out);
// 	f << "y;message" << endl;
// 	for(int i=0;i<(int)x_train.size();i++)
// 	{
// 		f << y_train[i] << ';';
// 		for(auto word : x_train[i])
// 			f << word << " ";
// 		f << endl;
// 	}
// 	f.close();
	
	unordered_set<string> stopwords;
	fstream stopfile("stopwords.txt",fstream::in);
	string stopword;
	while(stopfile >> stopword)
		stopwords.insert(stopword);
	
	sp_mat A = vectorizers::bin_vectorize(x_train);
	cout << size(A) << endl;
	
	int n_samples = A.n_rows,n = (n_samples*9)/10,p = A.n_cols,n_valid = n_samples - n;
	sp_mat X = A.head_rows(n);
	vector<double> y;
	for(bool x : y_train)
		y.push_back (x ? 1. : -1.);
	vec Y(vector<double>(y.begin(),y.begin() + n));
	clock_t start;
	
	start = clock();
	
	double seuil = 0.5; // we get 0.773756 accuracy. We do the same with scikit if we properly set class_set = class_weight={1 : np.sum(y_train)/n, 0 : 1 - np.sum(y_train)/n}
	double C = 1.,alpha = 0.002;
	vec w = zeros<vec>(p);
	// 	cout << "lolz" << endl;
	vec cur_grad = grad(X,Y,w,w0,C);
	// 	cout << "hey" << endl;
	double norm_grad = norm(cur_grad);
	double eps = 5e-2;
	int cmp = 0;
	cout << setprecision(6);
	cout << cur_grad.head(10) << endl;
	cout << val(X,Y,w,w0,C) << endl;
	cout << eps << " " << norm_grad << endl;
// 	while(norm_grad > eps)
// 	{
// 		++cmp;
// 		// 		cout << norm_grad << endl;
// 		// 		cout << cur_grad.second.head(10) << endl;
// 		w -= alpha*cur_grad;
// 		cur_grad = grad(X,Y,w,C);
// 		norm_grad = norm(cur_grad);
// 	}
// 	cout << cmp << endl;
// 	cout << "Done in " << ( clock() - start ) / (double) CLOCKS_PER_SEC << " secondes." << endl;
// 	sp_mat X_valid = A.tail_rows(n_samples - n);
// 	vec Y_valid(vector<double>(y.begin() + n,y.end()));
// 	// 	double x = 0.;
// 	// 	for(double b : y)
// 	// 		x += (b == -1.);
// 	// 	cout << x/n << endl;
// 	// 	
// 	vec Y_pred = 1./(1. + exp(-X_valid*w));
// 	// 	vec Y_pred = 1./(1. + exp(-X_valid*w));
// 	
// 	// 	auto Z = X_valid*w;
// 	// 	cout << Z.head_rows(10) << endl;
// 	// 	cout << Z.head(10) << endl;
// 	// 	cout << Y_pred.head(10) << endl;
// 	// 	cout << Y_validss.head(10) << endl;
// 	double score = 0.,score2 = 0;
// 	for(int i=0;i<(int)n_valid;i++)
// 	{
// 		score2 += Y_valid[i] == -1.;
// 		// 		if(Y_pred[i] > 0.5)
// 		// 			cout << i << endl;
// 		score += ((Y_pred[i] > seuil && Y_valid[i] == 1.) || (Y_pred[i] <= seuil && Y_valid[i] == -1.));
// 	}
// 	cout << score/n_valid << endl;
// 	
// 	cout << score2/n_valid << endl;
// 	// 	cout << norm_grad << endl;
// 	// 	cout << 
// 	// 	cout << cur_grad.second.head(10) << endl;
// 	//
	return 0;
}