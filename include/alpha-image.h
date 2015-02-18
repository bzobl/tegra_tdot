#ifndef ALPHA_IMAGE_H_INCLUDED
#define ALPHA_IMAGE_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class AlphaImage {

private:
  cv::Mat mColor;
  cv::Mat mAlpha;

  double mRatio;

public:
  AlphaImage(std::string filename);

  int width() const;
  int height() const;

  // scaled height, when width is given
  int height(int width) const;

  void write_scaled(cv::Mat &color, cv::Mat &alpha, cv::Rect targetROI) const;
};


#endif
