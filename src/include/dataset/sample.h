#ifndef FLATCTR_SAMPLE_H
#define FLATCTR_SAMPLE_H

#include <cstdint>
#include <memory>
#include <sstream>
#include <utility>
#include <vector>

#include "fast_float/fast_float.h"

#include "common.h"

using namespace std;

using SampleX = vector<pair<uint32_t, F>>;

class Sample
{
 private:
  static uint32_t fast_atoi(char** pptr);

 public:
  uint32_t            y;
  unique_ptr<SampleX> x;

  Sample(uint32_t y, unique_ptr<SampleX> x);
  explicit Sample(const string& line);

  [[nodiscard]] string to_string() const;

  ~Sample();
};

Sample::Sample(uint32_t y, unique_ptr<SampleX> x) : y(y), x(std::move(x)) {}

Sample::Sample(const string& line)
{
  char* p = const_cast<char*>(line.c_str());

  y = (*p++) - '0';

  x = make_unique<SampleX>();
  F val;
  do
  {
    p++;
    uint32_t idx = fast_atoi(&p);

    auto answer = fast_float::from_chars(p + 1, p + 100, val);
    x->push_back(make_pair(idx, val));
    p = const_cast<char*>(answer.ptr);
  } while (*(p) != '\0');
}

// https://stackoverflow.com/questions/16826422/c-most-efficient-way-to-convert-string-to-int-faster-than-atoi
uint32_t Sample::fast_atoi(char** pptr)
{
  char* p = *pptr;

  uint32_t val = 0;
  while (*p != ':')
  {
    val = val * 10 + (*(p++) - '0');
  }

  *pptr = p;
  return val;
}

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