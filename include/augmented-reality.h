#ifndef AUGMENTED_REALITY_H_INCLUDED
#define AUGMENTED_REALITY_H_INCLUDED

#include <atomic>
#include <cassert>

#include "opencv2/core.hpp"
#include "opencv2/cuda.hpp"

#include "alpha-image.h"
#include "faces.h"
#include "livestream.h"

class AugmentedReality {

private:
  LiveStream &mStream;
  Faces &mFaces;

  cv::cuda::CascadeClassifier_CUDA mFaceCascade;
  std::vector<AlphaImage> mHats;

public:

  AugmentedReality(LiveStream &stream, Faces &faces, std::string const &face_cascade);

  void addHat(std::string const &file);

  bool ready();
  void operator()();
};

#endif
