#ifndef ALPHA_IMAGE_H_INCLUDED
#define ALPHA_IMAGE_H_INCLUDED

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class AlphaImage {

private:
  cv::Mat mColor;
  cv::Mat mAlpha;

  double mRatio;
  double mToFaceWidthScale;
  double mToFaceOffset;

public:
  AlphaImage(std::string filename, double to_face_scale, double to_face_offset);

  int width() const;
  int height() const;

  // scaled width and height, when width of face is given
  int width(int face_width) const;
  int height(int face_width) const;
  int offset(int face_width) const;

  void write_scaled(cv::Mat &color, cv::Mat &alpha, cv::Rect targetROI) const;
};


#endif
