#include "alpha-image.h"

#include <iostream>

#include "opencv2/highgui/highgui.hpp"

AlphaImage::AlphaImage(std::string filename)
{
  cv::Mat image = cv::imread(filename, cv::IMREAD_UNCHANGED);
  mColor = cv::Mat(image.rows, image.cols, CV_8UC3);
  mAlpha = cv::Mat(image.rows, image.cols, CV_8UC1);
  cv::Mat out[] = {mColor, mAlpha};
  int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};
  cv::mixChannels(&image, 1, out, 2, from_to, 4);

  mRatio = (double)image.cols / (double)image.rows;

  std::cout << "loaded alphaimage '" << filename << "': " << image.cols << "x" << image.rows
            << "pixels (ratio " << mRatio << ")" << std::endl;
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

  //scale image
  cv::resize(mColor, scaled_color, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);
  cv::resize(mAlpha, scaled_alpha, scaled_size, 1.0, 1.0, cv::INTER_CUBIC);

  if (targetROI.x < 0) {
    // cut image left
    roi.x += std::abs(targetROI.x);
    roi.width -= std::abs(targetROI.x);
    targetROI.width -= std::abs(targetROI.x);
    targetROI.x = 0;
  }
  if (targetROI.x >= color.cols) {
    std::cerr << "target roi is not in image: " << targetROI << std::endl;
    return;
  } else if ((targetROI.x + targetROI.width) >= color.cols) {
    // cut image right
    int overlap = (targetROI.x + targetROI.width) - color.cols;
    roi.width -= overlap;
    targetROI.width -= overlap;
  }

  if (targetROI.y < 0) {
    // cut image top
    roi.y += std::abs(targetROI.y);
    roi.height -= std::abs(targetROI.y);
    targetROI.height -= std::abs(targetROI.y);
    targetROI.y = 0;
  }
  if (targetROI.y >= color.rows) {
    std::cerr << "target roi is not in image: " << targetROI << std::endl;
    return;
  } else if ((targetROI.y + targetROI.height) >= color.rows) {
    // cut image bottom
    int overlap = (targetROI.y + targetROI.height) - color.rows;
    roi.height -= overlap;
    targetROI.height -= overlap;
  }

  for (int x = 0; x < targetROI.width; x++) {
    for (int y = 0; y < targetROI.height; y++) {
      if (scaled_alpha(roi).at<uchar>(y, x) != 0) {
        alpha(targetROI).at<uchar>(y, x) = scaled_alpha(roi).at<uchar>(y, x);
        color(targetROI).at<cv::Vec3b>(y, x) = scaled_color(roi).at<cv::Vec3b>(y, x);
      }
    }
  }
}
