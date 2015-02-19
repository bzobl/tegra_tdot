#include "alpha-image.h"

#include <iostream>

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
  return mColor.cols / mRatio;
}

void AlphaImage::write_scaled(cv::Mat &color, cv::Mat &alpha, cv::Rect targetROI) const
{
  cv::Mat scaled_color, scaled_alpha;
  cv::Size scaled_size(targetROI.width, targetROI.height);
  cv::Rect roi(cv::Point(0, 0), scaled_size);

  std::cout << "New image to overlay on: " << targetROI << std::endl;

  //scale image
  cv::resize(mColor, scaled_color, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);
  cv::resize(mAlpha, scaled_alpha, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);

  if (targetROI.x < 0) {
    // cut image left
    std::cout << "cut image left" << std::endl;
    roi.x += std::abs(targetROI.x);
    roi.width -= std::abs(targetROI.x);
  }
  if ((targetROI.x + targetROI.width) >= color.cols) {
    // cut image right
    std::cout << "cut image right" << std::endl;
    roi.width -= (targetROI.x + targetROI.width) - color.cols;
  }
  if (targetROI.y < 0) {
    // cut image top
    std::cout << "cut image top" << std::endl;
    roi.y += std::abs(targetROI.y);
    roi.width -= std::abs(targetROI.y);
  }
  if ((targetROI.y + targetROI.height) >= color.rows) {
    std::cout << "cut image bottom" << std::endl;
    roi.width -= (targetROI.y + targetROI.height) - color.rows;
  }

  /*
  if (roi.x < 0 || roi.y < 0
      || (roi.x + roi.width) >= scaled_color.cols || (roi.y + roi.height) >= scaled_color.rows) {
    std::cout << "cannot draw roi: " << roi << std::endl;
    return;
  }
  */

  scaled_color(roi).copyTo(color(targetROI));
  scaled_alpha(roi).copyTo(alpha(targetROI));
}
