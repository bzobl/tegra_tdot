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

  // pointers to the GpuMats are used to allow fast swapping of last and new images
  cv::cuda::GpuMat mGpuImg1;
  cv::cuda::GpuMat mGpuImg2;
  cv::cuda::GpuMat *mNowGpuImg, *mLastGpuImg;

  void load_new_frame();
  void use_farneback(cv::Mat &flowx, cv::Mat &flowy, double &calc_time, double &dl_time);
  cv::Mat visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy);

public:
  OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization);

  bool isReady();
  void operator()();

};

#endif
