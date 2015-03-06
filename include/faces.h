#ifndef FACES_H_INCLUDED
#define FACES_H_INCLUDED

#include <mutex>

#include "opencv2/core.hpp"

class Faces {

private:
  struct FaceEntry {
    cv::Rect face;
    int ttl;
  };

  std::mutex mMutex;
  std::vector<FaceEntry> mFaces;

  int const DEFAULT_TTL = 3;

public:

  void addFace(cv::Rect &face);

  void tick();

  std::mutex &getMutex();
  std::vector<cv::Rect> getFaces();

};

#endif
