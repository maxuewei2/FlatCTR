#ifndef FLATCTR_FM_MODEL_H
#define FLATCTR_FM_MODEL_H

#include <cstdlib>
#include <immintrin.h>
#include <map>
#include <memory>
#include <unordered_map>

#include "libcuckoo/cuckoohash_map.hh"

#include "base_model.h"
#include "common.h"
#include "dataset/sample.h"

class FM_weight
{
 public:
  F              w = 0;
  std::vector<F> v;

  FM_weight() = default;

  void init(size_t N)
  {
    size_t n = ((N - 1) / 8 + 1) * 8; // align to 8
    w        = 0;
    v.clear();
    v.resize(n, 0);
    //fill(v.begin(), v.end(), 0);
  }

  explicit FM_weight(size_t N)
  {
    init(N);
  }
};

class FM : public Base
{
 private:
  size_t N;

  F w_lr;
  F v_lr;
  F w_l2;
  F v_l2;

  libcuckoo::cuckoohash_map<uint32_t, FM_weight> weights;
  F                                              bias = 0;

  std::default_random_engine  rand_generator;
  std::normal_distribution<F> gauss_distribution;

 public:
  FM(size_t N, F w_lr, F v_lr, F w_l2, F v_l2, F init_stddev, long seed);

  void learn(const std::vector<std::unique_ptr<Sample>>& sample_batch) override;

  F predict_prob(const std::unique_ptr<Sample>& sample, bool training);

  F predict_prob(const std::unique_ptr<Sample>& sample) override;

  void sgd(const F& bias_grad, const std::unordered_map<uint32_t, FM_weight>& grad_map);

  size_t load(const std::string& fname) override;

  int save(const std::string& fname) override;
};

FM::FM(size_t N, F w_lr, F v_lr, F w_l2, F v_l2, F init_stddev, long seed)
: N(N), w_lr(w_lr), v_lr(v_lr), w_l2(w_l2), v_l2(v_l2)
{
  if (seed != -1)
    rand_generator = std::default_random_engine(seed);
  gauss_distribution = std::normal_distribution<F>(0, init_stddev);
}

void FM::sgd(const F& bias_grad, const std::unordered_map<uint32_t, FM_weight>& grad_map)
{
  static thread_local FM_weight weight(N);

  for (auto& [idx, val] : grad_map)
  {
    weights.find(idx, weight);
    weight.w += (w_lr * val.w);
    for (size_t j = 0; j < N; j += 8)
    {
      __m256 v = _mm256_loadu_ps(weight.v.data() + j);
      __m256 g = _mm256_loadu_ps(val.v.data() + j);
      v        = _mm256_add_ps(v, _mm256_mul_ps(_mm256_set1_ps(v_lr), g));
      _mm256_storeu_ps(weight.v.data() + j, v);
    }
    weights.insert_or_assign(idx, weight);
  }

  bias += (w_lr * bias_grad);
}

void FM::learn(const std::vector<std::unique_ptr<Sample>>& sample_batch)
{
  static thread_local std::unordered_map<uint32_t, FM_weight> grad_map;
  static thread_local FM_weight                               weight(N);

  auto       size = (float)sample_batch.size();
  FM_weight* grad;
  F          bias_grad = 0;
  for (auto& sample : sample_batch)
  {
    uint32_t& y = sample->y;
    F         p = predict_prob(sample, true);
    F         t = (float)y - p;
    bias_grad += (t / size);

    for (size_t j = 0; j < N; j += 8)
    {
      __m256 sum_of_vx = _mm256_set1_ps(0);
      for (auto& [i, xi] : *(sample->x))
      {
        weights.find(i, weight);
        __m256 v  = _mm256_loadu_ps(weight.v.data() + j);
        __m256 x  = _mm256_set1_ps(xi);
        v         = _mm256_mul_ps(v, x);
        sum_of_vx = _mm256_add_ps(sum_of_vx, v);
      }
      for (auto& [i, xi] : *(sample->x))
      {
        weights.find(i, weight);
        if (grad_map.find(i) == grad_map.end())
        {
          grad = &grad_map[i];
          grad->init(N); // if i not in grad_map, make a zero grad
        }
        else
        {
          grad = &grad_map[i];
        }
        if (j == 0) [[unlikely]] // linear part
        {
          grad->w += (t * xi - w_l2 * weight.w) / size;
        }
        __m256 x   = _mm256_set1_ps(xi);
        __m256 tmp = _mm256_mul_ps(sum_of_vx, x);
        __m256 v   = _mm256_loadu_ps(weight.v.data() + j);
        x          = _mm256_mul_ps(x, x);
        x          = _mm256_mul_ps(x, v);
        x          = _mm256_sub_ps(tmp, x);
        x          = _mm256_mul_ps(x, _mm256_set1_ps(t));
        x          = _mm256_sub_ps(x, _mm256_mul_ps(_mm256_set1_ps(v_l2), v));
        __m256 g   = _mm256_loadu_ps(grad->v.data() + j);
        g          = _mm256_add_ps(g, _mm256_div_ps(x, _mm256_set1_ps(size)));
        _mm256_storeu_ps(grad->v.data() + j, g);
      }
    }
  }

  sgd(bias_grad, grad_map);
  grad_map.clear();
}

F FM::predict_prob(const std::unique_ptr<Sample>& sample)
{
  return predict_prob(sample, false);
}

F FM::predict_prob(const std::unique_ptr<Sample>& sample, bool training)
{
  static thread_local FM_weight weight(N);
  F                             p = bias;
  for (auto& [i, xi] : *(sample->x))
  {
    if (!weights.find(i, weight))
    {
      if (training)
      {
        weight.w = 0;
        for (size_t k = 0; k < N; k++)
          weight.v[k] = gauss_distribution(rand_generator);
        weights.insert(i, weight);
      }
      else
      {
        continue;
      }
    }
    p += (weight.w * xi);
  }

  __m256 res = _mm256_set1_ps(0);
  for (size_t j = 0; j < N; j += 8)
  {
    __m256 sum = _mm256_set1_ps(0), sum_of_square = _mm256_set1_ps(0);
    for (auto& [i, xi] : *(sample->x))
    {
      if (!weights.find(i, weight))
        if (!training)
          continue;
      __m256 v      = _mm256_loadu_ps(weight.v.data() + j);
      __m256 x      = _mm256_set1_ps(xi);
      v             = _mm256_mul_ps(v, x);
      sum           = _mm256_add_ps(sum, v);
      sum_of_square = _mm256_add_ps(sum_of_square, _mm256_mul_ps(v, v));
    }
    sum = _mm256_sub_ps(_mm256_mul_ps(sum, sum), sum_of_square);
    res = _mm256_add_ps(res, sum);
  }
  res = _mm256_hadd_ps(res, res);
  res = _mm256_hadd_ps(res, res);
  res = _mm256_hadd_ps(res, res);
  _mm256_storeu_ps(weight.v.data(), res);
  p += (0.5f * weight.v[0]);
  p = sigmoid(p);
  return p;
}

#define check_line(line, tokens, n)                                                                \
  if (tokens.size() != (n))                                                                        \
  {                                                                                                \
    spdlog::error("model parse error. @token, line: [{}] token_size: [{}]", line, tokens.size());  \
    return 0;                                                                                      \
  }
#define parse_idx(line, p, idx)                                                                    \
  errno = 0;                                                                                       \
  idx = strtol(p, &endptr, 10);                                                                    \
  if (errno != 0)                                                                                  \
  {                                                                                                \
    spdlog::error("model parse error. @idx,   line: [{}], p: [{}]", line, p);                      \
    return 0;                                                                                      \
  }
#define parse_val(line, p, val)                                                                    \
  answer = fast_float::from_chars(p, (p) + 100, val);                                              \
  if (answer.ec != std::errc())                                                                    \
  {                                                                                                \
    spdlog::error("model parse error. @val    line: [{}], p: [{}]", line, p);                      \
    return 0;                                                                                      \
  }
inline bool next_tokens(std::ifstream& ifs, std::string& line, std::vector<std::string>& tokens)
{
  if (!getline(ifs, line))
    return false;
  tokens.clear();
  string_split(line, tokens, "\t");
  return true;
}
size_t FM::load(const std::string& fname)
{
  std::ifstream::sync_with_stdio(false);
  std::ifstream ifs;
  ifs.open(fname, std::ifstream::in);

  std::string                   line;
  std::vector<std::string>      tokens;
  char*                         endptr;
  fast_float::from_chars_result answer{};

  next_tokens(ifs, line, tokens);
  check_line(line, tokens, 2);
  if (tokens[0] != "k")
  {
    spdlog::error("model parse error. @k");
    return 0;
  }
  parse_idx(line, tokens[1].c_str(), N);

  next_tokens(ifs, line, tokens);
  check_line(line, tokens, 2);
  if (tokens[0] != "bias")
  {
    spdlog::error("model parse error. @bias");
    return 0;
  }
  parse_val(line, tokens[1].c_str(), bias);

  FM_weight weight(N);
  uint32_t  idx;
  F         val;
  while (next_tokens(ifs, line, tokens))
  {
    check_line(line, tokens, N + 2);

    parse_idx(line, tokens[0].c_str(), idx);

    parse_val(line, tokens[1].c_str(), val);
    weight.w = val;

    for (size_t i = 0; i < N; ++i)
    {
      parse_val(line, tokens[i + 2].c_str(), val);
      weight.v[i] = val;
    }
    weights.insert(idx, weight);
  }
  ifs.close();
  return weights.size();
}

int FM::save(const std::string& fname)
{
  std::ofstream::sync_with_stdio(false);
  std::ofstream ofs;
  ofs.open(fname, std::ofstream::out);
  ofs << "k\t" << N << std::endl;
  ofs << "bias\t" << bias << std::endl;
  auto lt = weights.lock_table();
  for (const auto& it : lt)
  {
    const FM_weight& weight = it.second;
    ofs << it.first << "\t" << weight.w;
    for (size_t j = 0; j < N; ++j)
    {
      ofs << "\t" << weight.v[j];
    }
    ofs << std::endl;
  }
  ofs.close();
  return 0;
}

#endif //FLATCTR_FM_MODEL_H
