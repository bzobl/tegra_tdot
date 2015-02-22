#ifndef AUGMENTED_REALITY_H_INCLUDED
#define AUGMENTED_REALITY_H_INCLUDED

#include <atomic>
#include <cassert>

#include "opencv2/core.hpp"
#include "opencv2/cuda.hpp"

#include "alpha-image.h"
#include "livestream.h"

class AugmentedReality {

private:

  LiveStream &mStream;

  cv::cuda::CascadeClassifier_CUDA mFaceCascade;

  std::vector<AlphaImage> mHats;

  std::vector<cv::Rect> detect_faces(cv::Mat &frame);

public:

  AugmentedReality(LiveStream &stream, std::string const &face_cascade);

  void addHat(std::string const &file);

  bool ready();
  void operator()();
};

#endif
