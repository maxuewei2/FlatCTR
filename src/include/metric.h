#ifndef FLATCTR_METRIC_H
#define FLATCTR_METRIC_H

#include <algorithm>
#include <cassert>
#include <vector>

#include "common.h"

bool cmp_by_val(const std::pair<int, F>& a, const std::pair<int, F>& b)
{
  return (a.second < b.second);
}

double calc_auc(const std::vector<F>& y_pred, const std::vector<int>& y_true)
{
  assert(y_pred.size() == y_true.size() && "length of y_pred not equal to length of y_true");
  std::vector<std::pair<int, F>> lst;
  for (size_t i = 0; i < y_pred.size(); ++i)
  {
    lst.emplace_back(y_true[i], y_pred[i]);
  }
  std::sort(lst.begin(), lst.end(), cmp_by_val);
  double posNum = 0, posRankSum = 0, i = 1;
  for (auto item : lst)
  {
    if (item.first == 1)
    {
      posNum++;
      posRankSum += i;
    }
    i++;
  }
  return (posRankSum - (posNum * (posNum + 1) / 2.0)) / (posNum * ((F)y_pred.size() - posNum));
}

#endif //FLATCTR_METRIC_H
