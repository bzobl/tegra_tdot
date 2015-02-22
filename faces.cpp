#include "faces.h"

#include <algorithm>
#include <iostream>

#include "opencv2/imgproc.hpp"

Faces::Faces(std::string const &face_cascade) : mFaceCascade(face_cascade)
{
}

void Faces::addFace(cv::Rect &face)
{
  int separate_threshold = 20;

  for (auto &f : mFaces) {
    cv::Rect intersect = f.face & face;
    // found intersecting face -> update
    /*
    if (   (intersect.width > 0) && (intersect.width < separate_threshold)
        && (intersect.height > 0) && (intersect.height < separate_threshold)) {
    */
    if (intersect.width > 0) {
      f.face = face;
      f.ttl = DEFAULT_TTL;
      return;
    }
  }

  // no intersecting face found -> add new face
  std::cout << "adding new face " << face << std::endl;
  FaceEntry f = { face, DEFAULT_TTL };
  mFaces.emplace_back(f);
}

bool Faces::detect(cv::Mat const &frame)
{
  if (mFaceCascade.empty()) return false;

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

  return true;
}

void Faces::tick()
{
  for (auto &f : mFaces) {
    f.ttl--;
  }

  auto end = std::remove_if(mFaces.begin(), mFaces.end(),
                            [](FaceEntry const &f)
                            {
                              if (f.ttl <= 0) {
                               std::cout << "face " << f.face << "timed out ttl=" << f.ttl << std::endl;
                              }
                              return f.ttl <= 0;
                            });
  mFaces.erase(end, mFaces.end());
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
