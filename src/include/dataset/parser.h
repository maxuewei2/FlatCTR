#ifndef FLATCTR_PARSER_H
#define FLATCTR_PARSER_H

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>

#include "fast_float/fast_float.h"

#include "common.h"
#include "sample.h"

#define BUF_SIZE (256 * 1024 * 1024)

#define handle_error(msg)                                                                          \
  do                                                                                               \
  {                                                                                                \
    perror(msg);                                                                                   \
    exit(EXIT_FAILURE);                                                                            \
  } while (0)

using namespace std;

class Parser
{
 private:
  string file_name;
  int    fd;
  char*  buf;
  long   offset     = 0;
  long   bytes_read = 0;

  static uint32_t fast_atoi(char** pptr);
  void            read_block();

 public:
  explicit Parser(const string& file_name);

  ~Parser();

  void reset();

  static unique_ptr<Sample> parseSample(unique_ptr<string> line);

  unique_ptr<string> nextLine();
};

Parser::Parser(const string& file_name) : file_name(file_name)
{
  buf = new char[BUF_SIZE + 1];
  fd  = open(file_name.c_str(), O_RDONLY);
  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
  reset();
}

void Parser::reset()
{
  lseek(fd, 0, SEEK_SET);
  offset      = 0;
  buf[offset] = '\0';
  bytes_read  = 0;
}

Parser::~Parser()
{
  close(fd);
  delete buf;
}

// https://stackoverflow.com/questions/16826422/c-most-efficient-way-to-convert-string-to-int-faster-than-atoi
uint32_t Parser::fast_atoi(char** pptr)
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

unique_ptr<Sample> Parser::parseSample(unique_ptr<string> line)
{
  if (line == nullptr) [[unlikely]]
    return nullptr;

  char* p = const_cast<char*>(line->c_str());

  uint32_t y = (*p++) - '0';

  unique_ptr<SampleX> x = make_unique<SampleX>();
  F                   val;
  do
  {
    p++;
    uint32_t idx = fast_atoi(&p);

    auto answer = fast_float::from_chars(p + 1, p + 100, val);
    x->push_back(make_pair(idx, val));
    p = const_cast<char*>(answer.ptr);
  } while (*(p) != '\0');
  return make_unique<Sample>(y, std::move(x));
}

void Parser::read_block()
{
  bytes_read = read(fd, buf, BUF_SIZE);
  if (bytes_read == -1)
    handle_error("read failed");
  if (bytes_read)
  {
    long p = bytes_read - 1;
    while (buf[p] != '\n')
      p--;
    lseek(fd, -(bytes_read - p - 1), SEEK_CUR);
    bytes_read      = p + 1;
    buf[bytes_read] = '\n';
  }
}

unique_ptr<string> Parser::nextLine()
{
  if (offset >= bytes_read) [[unlikely]]
  {
    read_block();
    offset = 0;
    if (!bytes_read)
      return nullptr;
  }
  char* p                 = (char*)memchr(buf + offset, '\n', bytes_read - offset);
  *p                      = '\0';
  unique_ptr<string> line = make_unique<string>(buf + offset);
  offset                  = p - buf + 1;
  return line;
}

#endif