#include<iostream>
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

double val(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& rho)
{
	int n = X.n_rows;
	vec exp_vect = exp(-Y%(X*w + w0*ones<vec>(n)));
	return sum(log(1. + exp_vect))/n + rho*sum(w%w)/2.;
}

pair<double,vec> grad(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& rho)
{
	int n = X.n_rows;
// 	cout << "lolol" << endl;
	vec exp_vect = exp(-Y%(X*w + w0*ones<vec>(n)));
// 	cout << "yolo" << endl;
	vec coefs = -Y%exp_vect/(1. + exp_vect);
// 	cout << "merde" << endl;
	return {sum(coefs),(coefs.t()*X/n).t() + rho*w};
}

int main(void)
{
	cout << "YOLO" << endl;
	vector<vector<string> > x_train;
	vector<bool> y_train;
	vector<vector<string> > x_test;
	loaders::load_train("../data/train.csv",x_train,y_train);
	loaders::load_test("../data/test.csv",x_test);
	
	unordered_set<string> stopwords;
	fstream stopfile("stopwords.txt",fstream::in);
	string stopword;
	while(stopfile >> stopword)
		stopwords.insert(stopword);
	
	// It works, bitches.
// 	for(auto s : x_train[29])
// 		cout << s << endl;
// 	for(auto s : x_test[29])
// 		cout << s << endl;
	
// 	vector<unordered_map<string,int> > x_train2 = vectorizers::count_vectorize(x_train,stopwords);
	
// 	sparsem x_train2 = vectorizers::bin_vectorize(x_train);
// 	int n_features = x_train2.size2();
// 	bernouilli_naive_bayes clf(0.5);
// 	int taken_samples = 3800,n = x_train2.size1();
// 	sparsem x_train3 = boost::numeric::ublas::subrange(x_train2, 0,taken_samples, 0,n_features);
// 	sparsem x_valid = boost::numeric::ublas::subrange(x_train2, taken_samples,n, 0,n_features);
// 	
// 	vector<bool> y_train3(y_train.begin(),y_train.begin() + taken_samples);
// 	vector<bool> y_valid(y_train.begin() + taken_samples,y_train.end());
// 	clf.fit(x_train3,y_train3);
// 	std::vector<bool> y_pred = clf.predict(x_valid);
// 	cout << utils::score(y_pred,y_valid) << endl;
	
// 	vector<unordered_map<string,int> > x_train2 = vectorizers::n_gram_vectorize(x_train,grams);
// 	for(auto p : x_train2[29])
// 			cout << p.first << endl;
	
	////// Tentative of Naïve Bayes //////
	
	// Seems like for now the only way to make the classifier work better than random is by tuning alpha = 0...
	
// 	
// 	vector<unordered_map<string,int> > x_train3(x_train2.begin(),x_train2.begin() + taken_samples);
// 	
// 	vector<unordered_map<string,int> > x_valid(x_train2.begin() + taken_samples,x_train2.end());
// 	
// 	naive_bayes clf(0.5);
// 	clf.fit2(x_train3,y_train3);
// 	auto y_pred = clf.predict3(x_valid);
// 	
// 	cout << utils::score(y_pred,y_valid) << endl;
	
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
	
	double w0 = 0.,rho = 0.2,alpha = 0.002;
	vec w = zeros<vec>(p);
// 	cout << "lolz" << endl;
	auto cur_grad = grad(X,Y,w,w0,rho);
// 	cout << "hey" << endl;
	double norm_grad = sqrt(cur_grad.first*cur_grad.first + sum(cur_grad.second%cur_grad.second));
	double eps = 5e-2;
	while(norm_grad > eps)
	{
// 		cout << norm_grad << endl;
// 		cout << cur_grad.second.head(10) << endl;
		w0 -= alpha*cur_grad.first;
		w -= alpha*cur_grad.second;
		cur_grad = grad(X,Y,w,w0,rho);
		norm_grad = sqrt(cur_grad.first*cur_grad.first + sum(cur_grad.second%cur_grad.second));
	}
	cout << "Done in " << ( clock() - start ) / (double) CLOCKS_PER_SEC << " secondes." << endl;
	sp_mat X_valid = A.tail_rows(n_samples - n);
	vec Y_valid(vector<double>(y.begin() + n,y.end()));
// 	double x = 0.;
// 	for(double b : y)
// 		x += (b == -1.);
// 	cout << x/n << endl;
// 	
	vec Y_pred = 1./(1. + exp(- w0*ones<vec>(n_valid) - X_valid*w));
// 	vec Y_pred = 1./(1. + exp(-X_valid*w));
	
// 	auto Z = X_valid*w;
// 	cout << Z.head_rows(10) << endl;
// 	cout << Z.head(10) << endl;
// 	cout << Y_pred.head(10) << endl;
// 	cout << Y_validss.head(10) << endl;
	double score = 0.,score2 = 0;
	double seuil = 0.3111; // 0.3111 is near optimal
	// Le meilleur seuil est étrangement bas, j'ai du me tromper dans la fonction de prédiction.
	// En fait, c'est l'intercept qui cause ça. w0 ne fait que décaler le seuil de toute façon, il n'a pas de réel intérêt pour ce problème.
	for(int i=0;i<(int)n_valid;i++)
	{
		score2 += Y_valid[i] == -1.;
// 		if(Y_pred[i] > 0.5)
// 			cout << i << endl;
		score += ((Y_pred[i] > seuil && Y_valid[i] == 1.) || (Y_pred[i] <= seuil && Y_valid[i] == -1.));
	}
	cout << score/n_valid << endl;
	
	cout << score2/n_valid << endl;
// 	cout << norm_grad << endl;
// 	cout << 
// 	cout << cur_grad.second.head(10) << endl;
	//
	return 0;
}