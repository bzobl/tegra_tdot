#ifndef LIVESTREAM_H_INCLUDED
#define LIVESTREAM_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

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
  void applyOverlay(cv::Mat &image);
};

#endif
