#ifndef OPTICAL_FLOW_H_INCLUDED
#define OPTICAL_FLOW_H_INCLUDED

#include <map>

#include "opencv2/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"

#include "faces.h"
#include "livestream.h"
#include "thread-safe-mat.h"

class OpticalFlow {
private:
  LiveStream &mStream;

  ThreadSafeMat *mVisualizationImage;
  Faces *mFaces = nullptr;

  cv::cuda::FarnebackOpticalFlow mFarneback;

  // pointers to the GpuMats are used to allow fast swapping of last and new images
  cv::cuda::GpuMat mGpuImg1;
  cv::cuda::GpuMat mGpuImg2;
  cv::cuda::GpuMat *mNowGpuImg, *mLastGpuImg;

  static int const DIRECTION_UNDEFINED = 0;
  static int const DIRECTION_APPROACHING = 1;
  static int const DIRECTION_DISTANCING = 2;

  int get_direction_of_pixel(bool lower_half, cv::Point const &p1, cv::Point const & p2);

  void load_new_frame();
  void use_farneback(cv::Mat &flowx, cv::Mat &flowy, double &calc_time, double &dl_time);

  enum VisualizationType {
    OPTICAL_FLOW_VISUALIZATION_FACES = 0,
    OPTICAL_FLOW_VISUALIZATION_ARROWS = 1,
    OPTICAL_FLOW_VISUALIZATION_BLOCKS = 2,
    OPTICAL_FLOW_VISUALIZATION_LAST_ENTRY
  };

  static const std::map<OpticalFlow::VisualizationType, std::string> VISUALIZATION_NAMES; 

  VisualizationType mVisualization = OPTICAL_FLOW_VISUALIZATION_ARROWS;

  template <typename TFun>
  void visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy, TFun pixel_callback);
  cv::Mat visualize_optical_flow_arrows(cv::Mat const &flowx, cv::Mat const &flowy);
  cv::Mat visualize_optical_flow_blocks(cv::Mat const &flowx, cv::Mat const &flowy);
  cv::Mat visualize_optical_flow_faces(cv::Mat const &flowx, cv::Mat const &flowy);

public:
  OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization);

  bool isReady();
  void operator()();

  void setFaces(Faces *faces);
  void toggle_visualization();
};

#endif
