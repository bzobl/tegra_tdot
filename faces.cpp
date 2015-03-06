#include "faces.h"

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

void Faces::tick()
{
  std::unique_lock<std::mutex> l(mMutex);
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
