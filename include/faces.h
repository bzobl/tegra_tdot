#ifndef FACES_H_INCLUDED
#define FACES_H_INCLUDED

#include <mutex>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/cuda.hpp"

class Faces {

private:
  struct FaceEntry {
    cv::Rect face;
    int ttl;
  };

  cv::cuda::CascadeClassifier_CUDA mFaceCascade;

  std::mutex mMutex;
  std::vector<FaceEntry> mFaces;

  int const DEFAULT_TTL = 10;

  void addFace(cv::Rect &face);

public:
  Faces(std::string const &face_cascade);

  bool detect(cv::Mat const &frame);

  void tick();

  std::mutex &getMutex();
  std::vector<cv::Rect> getFaces();

};

#endif
