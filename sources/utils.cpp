#include"utils.hpp"

double utils::score(const std::vector<bool>& y_pred,const std::vector<bool>& y_valid)
{
	int pred_score = 0;
	for(int i=0;i<(int)y_pred.size();i++)
		pred_score += (y_pred[i] == y_valid[i]);
	return (double)pred_score/y_pred.size();
}