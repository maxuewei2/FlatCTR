#ifndef FLATCTR_UTIL_H
#define FLATCTR_UTIL_H

#include <cmath>
#include <random>

#include "common.h"

void string_split(const string& s, vector<string>& tokens, const string& delimiter)
{
  string::size_type last_pos = s.find_first_not_of(delimiter, 0);
  string::size_type pos      = s.find_first_of(delimiter, last_pos);
  while (string::npos != pos || string::npos != last_pos)
  {
    tokens.push_back(s.substr(last_pos, pos - last_pos));
    last_pos = s.find_first_not_of(delimiter, pos);
    pos      = s.find_first_of(delimiter, last_pos);
  }
}

#endif //FLATCTR_UTIL_H
