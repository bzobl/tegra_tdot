#ifndef LIVESTREAM_H_INCLUDED
#define LIVESTREAM_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "alpha-image.h"
#include <mutex>

class LiveStream {

private:
  cv::VideoCapture mCamera;
  int mStreamWidth = 0;
  int mStreamHeight = 0;

  cv::Mat mCurrentFrame;
  cv::Mat mOverlay;
  cv::Mat mOverlayAlpha;

  mutable std::mutex mFrameMutex;
  mutable std::mutex mOverlayMutex;

  bool openCamera(int num, int width, int height, int mode);

public:

  LiveStream(int camNum);
  LiveStream(int camNum, int width, int height);
  LiveStream(int camNum, int width, int height, int mode);

  virtual ~LiveStream();

  bool isOpened() const;

  void getFrame(cv::Mat &frame);
  void nextFrame(cv::Mat &frame);

  std::mutex &getOverlayMutex();
  void resetOverlay();
  void addImageToOverlay(AlphaImage const &image, int width, int x, int y);
  void applyOverlay(cv::Mat &image);
};

#endif
