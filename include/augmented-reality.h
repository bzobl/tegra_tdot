#ifndef AUGMENTED_REALITY_H_INCLUDED
#define AUGMENTED_REALITY_H_INCLUDED

#include <atomic>
#include <cassert>

#include "opencv2/core.hpp"

#include "alpha-image.h"
#include "faces.h"
#include "livestream.h"

class AugmentedReality {

private:
  LiveStream &mStream;
  Faces *mFaces;

  std::vector<AlphaImage> mHats;

public:
  AugmentedReality(LiveStream &stream, Faces *faces);

  void addHat(std::string const &file);

  bool ready();
  void operator()();
};

#endif
