#include<iostream>
#include"easySD.hpp"
#include<unordered_set>
#include<algorithm>
#include <armadillo>

using namespace std;
using namespace arma;

vector<int> grams = {1};

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
	
	SpMat<uchar> A = vectorizers::bin_vectorize(x_train);
	cout << size(A) << endl;
	//
	return 0;
}