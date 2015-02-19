#include "livestream.h"

#include <cassert>
#include <iostream>

LiveStream::LiveStream(int camNum) : LiveStream(camNum, -1, -1)
{
}

LiveStream::LiveStream(int camNum, int width, int height) : LiveStream(camNum, width, height, -1)
{
}

LiveStream::LiveStream(int camNum, int width, int height, int mode)
{
  if (!openCamera(camNum, width, height, mode)) {
    return;
  }

  mCamera.read(mCurrentFrame);
  mOverlay = cv::Mat::zeros(mStreamHeight, mStreamWidth, CV_8UC3);
  resetOverlay();
}

LiveStream::~LiveStream()
{
  if (mCamera.isOpened()) {
    mCamera.release();
  }
}

bool LiveStream::openCamera(int num, int width, int height, int mode)
{
  assert(num >= 0);
  mCamera.open(num);

  if (!mCamera.isOpened()) {
    std::cerr << "could not open camera" << std::endl;
    return false;
  }

  if ((width != -1) && (height != -1)) {
    if (   !mCamera.set(CV_CAP_PROP_FRAME_HEIGHT, height)
        || !mCamera.set(CV_CAP_PROP_FRAME_WIDTH, width)) {
      std::cerr << "could not set resolution " << width << "x" << height << std::endl;
      mCamera.release();
      return false;
    }
  }

  if (mode != -1) {
    if (!mCamera.set(CV_CAP_PROP_MODE, mode)) {
      std::cerr << "could not set mode " << mode << std::endl;
      mCamera.release();
      return false;
    }
  }

  mStreamWidth = mCamera.get(CV_CAP_PROP_FRAME_WIDTH);
  mStreamHeight = mCamera.get(CV_CAP_PROP_FRAME_HEIGHT);
  int m = mCamera.get(CV_CAP_PROP_MODE);

  std::cout << "initialized camera " << num << " with "
            << mStreamWidth << "x" << mStreamHeight
            << " Mode: " << m << std::endl;

  return true;
}

bool LiveStream::isOpened() const
{
  return mCamera.isOpened();
}

void LiveStream::getFrame(cv::Mat &frame)
{
  std::unique_lock<std::mutex> l(mFrameMutex);

  mCurrentFrame.copyTo(frame);
}

void LiveStream::nextFrame(cv::Mat &frame)
{
  std::unique_lock<std::mutex> l(mFrameMutex);

  mCamera.read(mCurrentFrame);
  mCurrentFrame.copyTo(frame);
}

std::mutex &LiveStream::getOverlayMutex()
{
  return mOverlayMutex;
}

void LiveStream::resetOverlay()
{
  mOverlayAlpha = cv::Mat::zeros(mStreamHeight, mStreamWidth, CV_8UC1);
}

void LiveStream::addImageToOverlay(AlphaImage const &image, int width, int x, int y)
{
  if (   (x < 0) || (x >= mStreamWidth)
      || (y < 0) || (y >= mStreamHeight)
      || (y + width < 0) || (x + width > mStreamWidth)) {
    std::cerr << "impossible location for image on overlay: (" << x << "," << y
              << ") widht=" << width << std::endl;
    return;
  }

  cv::Rect roi(x, y, width, image.height(width));
  image.write_scaled(mOverlay, mOverlayAlpha, roi);
}

void LiveStream::applyOverlay(cv::Mat &image)
{
  assert(image.cols == mStreamWidth);
  assert(image.rows == mStreamHeight);

  std::unique_lock<std::mutex> l(mOverlayMutex);

  for (int w = 0; w < mStreamWidth; w++) {
    for (int h = 0; h < mStreamHeight; h++) {
      if (mOverlayAlpha.at<uchar>(h, w) > 0) {
        image.at<cv::Vec3b>(h, w) = mOverlay.at<cv::Vec3b>(h, w);
      }
    }
  }
}
