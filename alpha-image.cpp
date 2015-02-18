#include "alpha-image.h"

#include "opencv2/highgui/highgui.hpp"

AlphaImage::AlphaImage(std::string filename)
{
  cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
  mColor = cv::Mat(image.rows, image.cols, CV_8UC3);
  mAlpha = cv::Mat(image.rows, image.cols, CV_8UC1);
  cv::Mat out[] = {mColor, mAlpha};
  int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};
  cv::mixChannels(&image, 1, out, 2, from_to, 4);

  mRatio = image.cols / image.rows;
}

int AlphaImage::width() const
{
  return mColor.cols;
}
int AlphaImage::height() const
{
  return mColor.rows;
}

int AlphaImage::height(int width) const
{
  return mColor.rows / mRatio;
}

void AlphaImage::write_scaled(cv::Mat &color, cv::Mat &alpha, cv::Rect targetROI) const
{
  cv::Mat scaled_color, scaled_alpha;
  cv::Size scaled_size(targetROI.width, targetROI.height);

  //scale image
  cv::resize(mColor, scaled_color, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);
  cv::resize(mAlpha, scaled_alpha, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);

  //TODO check if in range of image
  scaled_color.copyTo(color(targetROI));
  scaled_alpha.copyTo(alpha(targetROI));
}
