#ifndef FLATCTR_LR_MODEL_H
#define FLATCTR_LR_MODEL_H

#include <cstdlib>
#include <memory>
#include <unordered_map>

#include "libcuckoo/cuckoohash_map.hh"

#include "base_model.h"
#include "common.h"
#include "dataset/sample.h"
#include "util.h"

class LR : public Base
{
 private:
  F lr;
  F l2;

  libcuckoo::cuckoohash_map<uint32_t, F> weights;
  F                                      bias = 0;

 public:
  LR(F lr, F l2);

  void learn(const std::vector<std::unique_ptr<Sample>>& sample_batch) override;

  F predict_prob(const std::unique_ptr<Sample>& sample) override;

  size_t load(const std::string& fname) override;

  int save(const std::string& fname) override;
};

LR::LR(F lr, F l2) : lr(lr), l2(l2) {}

void LR::learn(const std::vector<std::unique_ptr<Sample>>& sample_batch)
{
  static thread_local std::unordered_map<uint32_t, F> grad_map;

  F bias_grad = 0;
  for (auto& sample : sample_batch)
  {
    uint32_t& y = sample->y;
    F         p = predict_prob(sample);
    F         t = (float)y - p;
    for (auto& [i, xi] : *(sample->x))
    {
      F w = 0;
      weights.find(i, w);
      if (grad_map.find(i) == grad_map.end())
        grad_map[i] = 0;
      grad_map[i] += (t * xi - l2 * w);
    }
    bias_grad += t;
  }
  auto size = (float)sample_batch.size();
  {
    for (auto& [idx, val] : grad_map)
    {
      F w = 0;
      weights.find(idx, w);
      w += (lr * val / size);
      weights.insert_or_assign(idx, w);
    }
  }
  bias += (lr * bias_grad / size);
  grad_map.clear();
}

F LR::predict_prob(const std::unique_ptr<Sample>& sample)
{
  F p = bias;
  for (auto& [i, xi] : *(sample->x))
  {
    F w = 0;
    weights.find(i, w);
    p += (w * xi);
  }
  p = sigmoid(p);
  return p;
}

size_t LR::load(const std::string& fname)
{
  std::ifstream::sync_with_stdio(false);
  std::ifstream ifs;
  ifs.open(fname, std::ifstream::in);
  std::string tmp;
  ifs >> tmp >> bias;
  if (tmp != "bias")
  {
    spdlog::error("model parse error.");
    return 0;
  }
  uint32_t idx;
  F        val;
  while (!ifs.eof())
  {
    ifs >> idx >> val;
    weights.insert(idx, val);
  }
  ifs.close();
  return weights.size();
}

int LR::save(const std::string& fname)
{
  std::ofstream::sync_with_stdio(false);
  std::ofstream ofs;
  ofs.open(fname, std::ofstream::out);
  ofs << "bias\t" << bias << std::endl;
  auto lt = weights.lock_table();
  for (const auto& it : lt)
    ofs << it.first << "\t" << it.second << std::endl;
  ofs.close();
  return 0;
}

#endif