#ifndef FLATCTR_PARSER_H
#define FLATCTR_PARSER_H

#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <memory>
#include <unistd.h>

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

  void read_block();

 public:
  explicit Parser(const string& file_name);

  ~Parser();

  void reset();

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