#ifndef LOGREG_L
#define LOGREG_L

#include<armadillo>

using namespace arma;
using namespace std;

class logreg
{
private:
	double g(const sp_mat&,const vec&,const vec&,const double&,const double&);
	pair<double,vec> gradg(const sp_mat&,const vec&,const vec&,const double&,const double&);
	double norm(const pair<double,vec>&);
public:
	bool verbose = 0;
	double w0,C,tol,alpha,correct_rate;
	vec w;
	logreg(const double&,const double&,const double&,const double&);
	void fit(const sp_mat&,const vec&);
	uvec predict(const sp_mat&);
	double score(const sp_mat&,const vec&);
};

#endif