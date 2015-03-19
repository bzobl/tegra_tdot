#include "augmented-reality.h"

AugmentedReality::AugmentedReality(LiveStream &stream, Faces *faces)
                                  : mStream(stream), mFaces(faces)
{
}

void AugmentedReality::addHat(std::string const &file, double width_scale, double x_offset_scale)
{
  mHats.emplace_back(file, width_scale, x_offset_scale);
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

  int hat_idx = 0;

  for (cv::Rect face : mFaces->getFaces()) {
    int idx = hat_idx++ % mHats.size();
    AlphaImage &hat(mHats[idx]); 

    int width = hat.width(face.width);
    int x = face.x - hat.offset(face.width);
    int y = face.y - hat.height(face.width);
    mStream.addImageToOverlay(hat, face.width, x, y);
  }
}
