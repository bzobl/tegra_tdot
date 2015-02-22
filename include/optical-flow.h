#ifndef OPTICAL_FLOW_H_INCLUDED
#define OPTICAL_FLOW_H_INCLUDED

#include "opencv2/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"

#include "livestream.h"
#include "thread-safe-mat.h"

class OpticalFlow {

private:
  LiveStream &mStream;

  ThreadSafeMat &mVisualizationImage;

  cv::cuda::FarnebackOpticalFlow mFarneback;

  cv::cuda::GpuMat mGpuImg1;
  cv::cuda::GpuMat mGpuImg2;
  cv::cuda::GpuMat *mNowGpuImg, *mLastGpuImg;

  void use_farneback(cv::Mat &flowx, cv::Mat &flowy, double &calc_time, double &dl_time);
  void visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy, cv::Mat &result);

public:
  OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization);

  bool isReady();
  void operator()();

};

#endif
