#ifndef LIVESTREAM_H_INCLUDED
#define LIVESTREAM_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "alpha-image.h"

class LiveStream {

private:
  cv::VideoCapture mCamera;
  int mStreamWidth = 0;
  int mStreamHeight = 0;


  cv::Mat mCurrentFrame;
  cv::Mat mOverlay;
  cv::Mat mOverlayAlpha;

public:

  LiveStream(int camNum);
  virtual ~LiveStream();

  bool isOpened() const;

  void getFrame(cv::Mat &frame);
  void nextFrame(cv::Mat &frame);

  void resetOverlay();
  void writeOverlayImage(AlphaImage const &image, int width, int x, int y);
  void applyOverlay(cv::Mat &image);
};

#endif
