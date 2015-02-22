#ifndef THREAD_SAFE_MAT_INCLUDED
#define THREAD_SAFE_MAT_INCLUDED

#include <mutex>
#include "opencv2/core.hpp"

class ThreadSafeMat {

private:
  cv::Mat mMat;

  std::mutex mMutex;

public:
  ThreadSafeMat();
  ThreadSafeMat(cv::Mat mat);

  cv::Mat get();
  void update(cv::Mat new_mat);

};

#endif
