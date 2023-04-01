#ifndef FLATCTR_SAMPLE_H
#define FLATCTR_SAMPLE_H

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "common.h"

using namespace std;

using SampleX = vector<pair<uint32_t, F>>;

class Sample
{
 public:
  uint32_t            y;
  unique_ptr<SampleX> x;

  Sample(uint32_t y, unique_ptr<SampleX> x);

  [[nodiscard]] string to_string() const;

  ~Sample();
};

Sample::Sample(uint32_t y, unique_ptr<SampleX> x) : y(y), x(move(x)) {}

string Sample::to_string() const
{
  static thread_local stringstream sstream;
  sstream.str(string());
  sstream << y;
  for (auto& [i, val] : *x)
  {
    sstream << " " << i << ":" << val;
  }
  return sstream.str();
}

Sample::~Sample() = default;

#endif