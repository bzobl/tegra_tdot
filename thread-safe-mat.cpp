#include "thread-safe-mat.h"

ThreadSafeMat::ThreadSafeMat()
{
}

ThreadSafeMat::ThreadSafeMat(cv::Mat mat)
{
  mat.copyTo(mMat);
}

cv::Mat ThreadSafeMat::get()
{
  std::unique_lock<std::mutex> l(mMutex);
  cv::Mat ret;
  mMat.copyTo(ret);
  return ret;
}
void ThreadSafeMat::update(cv::Mat new_mat)
{
  std::unique_lock<std::mutex> l(mMutex);
  new_mat.copyTo(mMat);
}
