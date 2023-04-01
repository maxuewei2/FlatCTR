#ifndef FLATCTR_BASE_MODEL_H
#define FLATCTR_BASE_MODEL_H

#include "common.h"
#include "dataset/sample.h"

inline F sigmoid(F t)
{
  if (t < 0)
    return std::exp(t) / (1 + std::exp(t));
  return 1.0f / (1.0f + std::exp(-t));
}

class Base
{
 private:
 public:
  virtual void learn(const std::vector<std::unique_ptr<Sample>>& sample_batch) = 0;

  virtual F predict_prob(const std::unique_ptr<Sample>& sample) = 0;

  virtual size_t load(const std::string& fname) = 0;

  virtual int save(const std::string& fname) = 0;
};

#endif //FLATCTR_BASE_MODEL_H
