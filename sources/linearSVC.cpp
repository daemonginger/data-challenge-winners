#include"logreg.hpp"

logreg::logreg(const double& _C = 1.,const double& _tol = 1e-3,const double& _alpha = 0.002,const double& _correct_rate = 0.1):C(_C),tol(_tol),alpha(_alpha),correct_rate(_correct_rate) {}

double logreg::g(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	vec exp_vect = exp(-Y%(X*w + w0));
	return C*sum(log(1. + exp_vect)) + (sum(w%w) + w0*w0)/2.;
}

pair<double,vec> logreg::gradg(const sp_mat& X,const vec& Y,const vec& w,const double& w0,const double& C)
{
	vec exp_vect = exp(-Y%(X*w + w0));
	vec coefs = -Y%exp_vect/(1. + exp_vect);
	return {C*sum(coefs) + w0,C*((coefs.t()*X).t()) + w};
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
	double norm_grad = 0.,old_norm = 0.;
	
	vec Y = y;
	for(int i=0;i<n;i++)
		if(Y(i) == 0.)
			Y(i) = -1.;
		int iters = 0;
	do
	{
		++iters;
		if(verbose)
			cout << norm_grad << endl;
		old_norm = norm_grad;
		cur_grad = gradg(X,Y,w,w0,C);
		norm_grad = norm(cur_grad);
		w0 -= alpha*cur_grad.first;
		w -= alpha*cur_grad.second;
		// This is a dirty trick but is seems to work?
		if(iters != 1)
			alpha *= (norm_grad > old_norm ? -1 : 1)*correct_rate + 1.;
		
	}while(norm_grad > tol);
	
	if(verbose)
	{
		cout << "Objective function value : " << g(X,Y,w,w0,C) << endl;
		cout << "Iterations : " << iters << endl;
	}
}

uvec logreg::predict(const sp_mat& X)
{
	return ((X*w + w0) > 0.);
}

double logreg::score(const sp_mat& X,const vec& Y)
{
	return (double)(sum(predict(X) == (Y == 1.)))/Y.n_rows;
}