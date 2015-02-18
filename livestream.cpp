#include "livestream.h"

LiveStream::LiveStream(int camNum)
{
  assert(camNum >= 0);
  mCamera.open(camNum);

  if (!mCamera.isOpened()) {
    return;
  }

  mStreamWidth = mCamera.get(CV_CAP_PROP_FRAME_WIDTH);
  mStreamHeight = mCamera.get(CV_CAP_PROP_FRAME_HEIGHT);

  mCamera.read(mCurrentFrame);
  mOverlay = cv::Mat::zeros(mStreamHeight, mStreamWidth, CV_8UC3);
}

LiveStream::~LiveStream()
{
  if (mCamera.isOpened()) {
    mCamera.release();
  }
}

void LiveStream::resetOverlay()
{
  mOverlayAlpha = cv::Mat::zeros(mStreamHeight, mStreamWidth, CV_8UC1);
}

bool LiveStream::isOpened() const
{
  return mCamera.isOpened();
}

void LiveStream::getFrame(cv::Mat &frame)
{
  mCurrentFrame.copyTo(frame);
}

void LiveStream::nextFrame(cv::Mat &frame)
{
  mCamera >> mCurrentFrame;
  mCurrentFrame.copyTo(frame);
}

void LiveStream::applyOverlay(cv::Mat &image)
{
  for (int w = 0; w < mStreamWidth; w++) {
    for (int h = 0; h < mStreamHeight; h++) {
      if (mOverlayAlpha.at<uchar>(h, w) > 0) {
        image.at<cv::Vec3b>(h, w) = mOverlay.at<cv::Vec3b>(h, w);
      }
    }
  }
}
