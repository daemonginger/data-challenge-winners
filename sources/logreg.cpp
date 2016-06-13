#include"logreg.hpp"

logreg::logreg(const double& _C = 1.,const double& _tol = 1e-3,const double& _alpha = 0.002):C(_C),tol(_tol),alpha(_alpha) {}

double logreg::g(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	vec exp_vect = exp(-Y%(X*w + w0));
	return C*sum(log(1. + exp_vect)) + (sum(w%w) + w0*w0)/2.;
}

pair<double,vec> logreg::gradg(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	vec exp_vect = exp(-Y%(X*w + w0));
	vec coefs = -Y%exp_vect/(1. + exp_vect);
	return {sum(coefs) + w0,C*((coefs.t()*X).t()) + w};
}

double logreg::norm(const pair<double,vec>& gr)
{
	return sqrt(gr.first*gr.first + sum(gr.second%gr.second));
}

void logreg::fit(const sp_mat& X,const vec& y)
{
	int n = X.n_rows;
	int p = X.n_cols;
	w0 = 0.;
	w = zeros<vec>(p);
	pair<double,vec> cur_grad;
	double norm_grad = 0.;
	
	vec Y = y;
	for(int i=0;i<n;i++)
		if(Y[i] == 0.)
			Y[i] = -1.;
	
	do
	{
		cur_grad = gradg(X,Y,w,w0,C);
		norm_grad = norm(cur_grad);
		w0 -= alpha*cur_grad.first;
		w -= alpha*cur_grad.second;
	}while(norm_grad > tol);
}

uvec logreg::predict(const sp_mat& X)
{
	return ((X*w + w0) > 0.);
}

double logreg::score(const sp_mat& X,const vec& Y)
{
	return (double)(sum(predict(X) == (Y == 1.)))/Y.n_rows;
}