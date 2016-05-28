#include<iostream>
#include"easySD.hpp"

using namespace std;

int main(void)
{
	cout << "YOLO" << endl;
	vector<vector<string> > x_train;
	vector<bool> y_train;
	vector<vector<string> > x_test;
	loaders::load_train("../data/train.csv",x_train,y_train);
	loaders::load_test("../data/test.csv",x_test);
	
	// It works, bitches.
// 	for(auto s : x_train[29])
// 		cout << s << endl;
// 	for(auto s : x_test[29])
// 		cout << s << endl;
	
	
	
	return 0;
}