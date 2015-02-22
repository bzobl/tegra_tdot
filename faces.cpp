#include "faces.h"

#include <algorithm>

void Faces::addFace(cv::Rect *face)
{
  for (auto &f : mFaces) {
    // found intersecting face -> update
    if ((f.face & face).width > 0) {
      f.face = face;
      f.ttl = DEFAULT_TTL;
    }
  }

  // no intersecting face found -> add new face
  mFaces.emplace_back(face, DEFAULT_TTL);
}

void Faces::detect(cv::Mat const &frame)
{
  std::unique_lock<std::mutex> l(mMutex);

  std::vector<cv::Rect> faces;
  cv::cuda::GpuMat d_faces;
  cv::Mat h_faces;

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::cuda::GpuMat d_gray(gray);

  int n_detected = mFaceCascade.detectMultiScale(d_gray, d_faces, 1.2, 8, cv::Size(40, 40));

  d_faces.colRange(0, n_detected).download(h_faces);
  cv::Rect *prect = h_faces.ptr<cv::Rect>();

  for (int i = 0; i < n_detected; i++) {
    addFace(prect[i]);
  }
}

void Faces::tick()
{
  for (auto &f : mFaces) {
    f.ttl--;
  }

  std::remove_if(mFaces.begin(), mFaces.end(),
                 [](FaceEntry const &f) { return f.ttl == 0; });
}

std::mutex &Faces::getMutex()
{
  return mMutex;
}

std::vector<cv::Rect> Faces::getFaces()
{
  std::vector<cv::Rect> faces;
  for (auto &f : mFaces) {
    faces.push_back(f.face);
  }
  return faces;
}
