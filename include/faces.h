#ifndef FACES_H_INCLUDED
#define FACES_H_INCLUDED

#include <mutex>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/cuda.hpp"

class Faces {

private:
  std::mutex mMutex;
  std::vector<cv::Rect> mFaces;

  int const DEFAULT_TTL = 10;

  struct FaceEntry {
    cv::Rect face;
    int ttl;
  };

  void addFace(cv::Rect *face);

public:
  void detect(cv::Mat const &frame);

  void tick();

  std::mutex &getMutex();
  std::vector<cv::Rect> &getFaces();

};

#endif
