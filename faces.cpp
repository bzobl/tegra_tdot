#include "faces.h"

#include <algorithm>
#include <cassert>
#include <iostream>

#include "opencv2/imgproc.hpp"

Faces::Faces(LiveStream &stream, std::string const &face_cascade) : mStream(stream),
                                                                    mFaceCascade(face_cascade)
{
}

void Faces::addFace(cv::Rect &face)
{
  for (auto &f : mFaces) {
    cv::Rect intersect = f.face & face;
    if (intersect.width > 0) {
      f.face = face;
      f.ttl = DEFAULT_TTL;
      return;
    }
  }

  // no intersecting face found -> add new face
  FaceEntry f = { face, DEFAULT_TTL };
  mFaces.emplace_back(f);
}

bool Faces::isReady()
{
  return mStream.isOpened() && !mFaceCascade.empty();
}

bool Faces::detect()
{
  assert(isReady());

  cv::Mat frame, h_faces;
  cv::cuda::GpuMat d_frame, d_faces;
  std::vector<cv::Rect> faces;

  std::unique_lock<std::mutex> l(mMutex);

  // update ttl of all faces
  tick();

  mStream.getFrame(frame);
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
  d_frame.upload(frame);

  int n_detected = mFaceCascade.detectMultiScale(d_frame, d_faces, 1.2, 8, cv::Size(40, 40));

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

  auto end = std::remove_if(mFaces.begin(), mFaces.end(),
                            [](FaceEntry const &f) { return f.ttl <= 0; });
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
