#include "augmented-reality.h"

AugmentedReality::AugmentedReality(LiveStream &stream, Faces *faces)
                                  : mStream(stream), mFaces(faces)
{
}

void AugmentedReality::addHat(std::string const &file)
{
  mHats.emplace_back(file);
}

bool AugmentedReality::ready()
{
  return mStream.isOpened() && !mHats.empty();
}

void AugmentedReality::operator()()
{
  assert(ready());

  // for the duration of resetting the overlay no other thread must use the overlay
  std::unique_lock<std::recursive_mutex> sl(mStream.getOverlayMutex(), std::defer_lock);
  std::unique_lock<std::mutex> fl(mFaces->getMutex(), std::defer_lock);
  lock(sl, fl);

  mStream.resetOverlay();

  for (cv::Rect face : mFaces->getFaces()) {
    AlphaImage &hat(mHats[0]); 

    int width = face.width * 2;
    int height = hat.height(width);
    int x = face.x - width/4;
    int y = face.y - height;
    mStream.addImageToOverlay(hat, width, x, y);
  }
}
