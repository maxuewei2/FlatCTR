#ifndef _FLATCTR_BLOCKING_QUEUE_H_
#define _FLATCTR_BLOCKING_QUEUE_H_

#include <condition_variable>
#include <deque>

using namespace std;

template <typename T>
class BlockingQueue
{
 private:
  deque<T>           queue;
  size_t             n;
  mutex              mtx;
  condition_variable not_empty;
  condition_variable not_full;

 public:
  explicit BlockingQueue(size_t n) : n(n) {}

  void push(T&& t)
  {
    unique_lock<mutex> lck(mtx);
    while (queue.size() > n)
      not_full.wait(lck);
    queue.push_back(std::move(t));
    not_empty.notify_one();
    lck.unlock();
  }

  void pop(T& t)
  {
    unique_lock<mutex> lck(mtx);
    while (queue.empty())
      not_empty.wait(lck);
    t = std::move(queue.front());
    queue.pop_front();
    not_full.notify_one();
    lck.unlock();
  }
};

#endif //_FLATCTR_BLOCKING_QUEUE_H_
