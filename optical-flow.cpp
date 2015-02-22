#include "optical-flow.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <sstream>

OpticalFlow::OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization)
                        : mStream(stream), mVisualizationImage(&visualization)
{
  mNowGpuImg = &mGpuImg1;
  mLastGpuImg = &mGpuImg2;
  load_new_frame();

  mFarneback.numLevels = 5;         // number of pyramid layers including initial
  mFarneback.pyrScale = 0.5;        // scale for pyramids. 0.5: next layer is twice smaller
  mFarneback.fastPyramids = false;
  mFarneback.winSize = 13;          // averaging window size
  mFarneback.numIters = 1;         // iterations per pyramid level
  mFarneback.polyN = 5;             // size of pixel neighborhood. usally 5 or 7
  mFarneback.polySigma = 1.1;       // standard deviation for gaussian usually 1.1 or 1.5
  mFarneback.flags = 0;
}

bool OpticalFlow::isReady()
{
  return mStream.isOpened();
}

int OpticalFlow::get_direction_of_pixel(bool lower_half, cv::Point const &p1, cv::Point const & p2)
{
  double const diff_threshold = 1;

  double diff = p1.y - p2.y;

  if (lower_half) {
  // lower half -> arrow pointing down when approaching
    if (diff < (diff_threshold * -1)) {
      return DIRECTION_APPROACHING;
    } else if (diff > diff_threshold) {
      return DIRECTION_DISTANCING;
    } else {
      return DIRECTION_UNDEFINED;
    }
  } else {
  // upper half -> arrow pointing up when approaching
    if (diff < (diff_threshold * -1)) {
      return DIRECTION_DISTANCING;
    } else if (diff > diff_threshold) {
      return DIRECTION_APPROACHING;
    } else {
      return DIRECTION_UNDEFINED;
    }
  }
}

void OpticalFlow::load_new_frame()
{
  cv::Mat image;

  // swap pointers to avoid reallocating memory on gpu
  std::swap(mNowGpuImg, mLastGpuImg);

  mStream.getFrame(image);
  cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  mNowGpuImg->upload(image);
}

void OpticalFlow::use_farneback(cv::Mat &flowx, cv::Mat &flowy,
                                double &calc_time_ms, double &dl_time_ms)
{
  cv::cuda::GpuMat d_flowx, d_flowy;

  double calc_start = (double) cv::getTickCount();
  mFarneback(*mLastGpuImg, *mNowGpuImg, d_flowx, d_flowy);
  calc_time_ms = ((double) cv::getTickCount() - calc_start) / cv::getTickFrequency() * 1000;

  double dl_start = (double) cv::getTickCount();
  d_flowx.download(flowx);
  d_flowy.download(flowy);
  dl_time_ms = ((double) cv::getTickCount() - dl_start) / cv::getTickFrequency() * 1000;
}

template <typename TFun>
void OpticalFlow::visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy,
                                         TFun pixel_callback)
{
  int const width = flowx.cols;
  int const height = flowx.rows;
  double const l_threshold = 2;

  for (int y = 0; y < height; y += 10) {
    for (int x = 0; x < width; x += 10) {
      double dx = flowx.at<float>(y, x);
      double dy = flowy.at<float>(y, x);

      double l = std::sqrt(dx*dx + dy*dy);

      if ((l > l_threshold)) {
        cv::Point p(x, y);
        cv::Point p2(x + dx, y + dy);
        int direction = get_direction_of_pixel((y > height/2), p, p2);

        pixel_callback(p, p2, direction);
      }
    }
  }
}

cv::Mat OpticalFlow::visualize_optical_flow_blocks(cv::Mat const &flowx, cv::Mat const &flowy)
{
  cv::Mat result = cv::Mat::zeros(flowx.rows, flowy.cols, CV_8UC3);;
  cv::Mat directions = cv::Mat::zeros(flowx.rows, flowy.cols, CV_8UC1);;

  visualize_optical_flow(flowx, flowy,
                         [&directions](cv::Point const &p1, cv::Point const &p2, unsigned char direction)
                         {
                          directions.at<uchar>(p1.y, p1.x) = direction;
                         });

  int const n_xblocks = 15;
  int const n_yblocks = 15;
  int const x_pixels_per_block = mStream.width() / n_xblocks;
  int const y_pixels_per_block = mStream.height() / n_yblocks;

  for (int x = 0 ; x < n_xblocks; x++) {
    for (int y = 0 ; y < n_yblocks; y++) {
      int width = (x != n_xblocks - 1) ? x_pixels_per_block
                                       : (mStream.width() - x * x_pixels_per_block);
      int height = (y != n_yblocks - 1) ? y_pixels_per_block
                                        : (mStream.height() - y * y_pixels_per_block);
      cv::Rect roi(x * x_pixels_per_block, y * y_pixels_per_block, width, height);

      int sum_approaching = std::count_if(directions(roi).begin<uchar>(),
                                          directions(roi).end<uchar>(),
                                          [](unsigned char v)
                                          {
                                            return v == DIRECTION_APPROACHING;
                                          });
      int sum_distancing = std::count_if(directions(roi).begin<uchar>(),
                                         directions(roi).end<uchar>(),
                                         [](unsigned char v)
                                         {
                                           return v == DIRECTION_DISTANCING;
                                         });
      int sum_undefined = std::count_if(directions(roi).begin<uchar>(),
                                        directions(roi).end<uchar>(),
                                        [](unsigned char v)
                                        {
                                          return (v != DIRECTION_APPROACHING) && (v != DIRECTION_DISTANCING);
                                        });

      int block_direction;
      if (sum_approaching > sum_distancing) {
        if (sum_undefined > sum_approaching) {
          block_direction = DIRECTION_UNDEFINED;
        } else {
          block_direction = DIRECTION_APPROACHING;
        }
      } else {
        if (sum_undefined > sum_distancing) {
          block_direction = DIRECTION_UNDEFINED;
        } else {
          block_direction = DIRECTION_DISTANCING;
        }
      }

      cv::Scalar color;
      switch (block_direction) {
        case DIRECTION_APPROACHING:
          color = cv::Scalar(0, 255, 0);
          break;
        case DIRECTION_DISTANCING:
          color = cv::Scalar(0, 0, 255);
          break;
        default:
          color = cv::Scalar(255, 255, 255);
          break;
      }

      cv::rectangle(result, roi, color, cv::CV_FILLED);
    }
  }

  return result;
}

cv::Mat OpticalFlow::visualize_optical_flow_arrows(cv::Mat const &flowx, cv::Mat const &flowy)
{
  cv::Mat result = cv::Mat::zeros(flowx.rows, flowy.cols, CV_8UC3);;

  visualize_optical_flow(flowx, flowy,
                         [&result](cv::Point const &p1, cv::Point const &p2, unsigned char direction)
                         {
                          cv::Scalar color;
                          switch (direction) {
                            case DIRECTION_APPROACHING:
                              color = cv::Scalar(0, 255, 0);
                              break;
                            case DIRECTION_DISTANCING:
                              color = cv::Scalar(0, 0, 255);
                              break;
                            default:
                              color = cv::Scalar(255, 255, 255);
                              break;
                          }
                          cv::arrowedLine(result, p1, p2, color);
                         });
  return result;
}

void OpticalFlow::operator()()
{
  assert(isReady());

  cv::Mat flowx, flowy, result;

  double ul_start = (double) cv::getTickCount();
  load_new_frame();
  double ul_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  double calc_time, dl_time;
  use_farneback(flowx, flowy, calc_time, dl_time);

  double visualize_start = (double) cv::getTickCount();
  switch (mVisualization) {
    case OPTICAL_FLOW_VISUALIZATION_ARROWS:
      result = visualize_optical_flow_arrows(flowx, flowy);
      break;
    case OPTICAL_FLOW_VISUALIZATION_BLOCKS:
      result = visualize_optical_flow_blocks(flowx, flowy);
      break;
  }
  double visualize_time_ms = ((double) cv::getTickCount() - visualize_start) / cv::getTickFrequency() * 1000;

  double total_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  {
    cv::Point pos(50, 50);
    int font = cv::FONT_HERSHEY_PLAIN;
    cv::Scalar color(255, 255, 255);
    double scale = 1.2;

    struct {
      std::string text;
      double *time;
    } times[] = {
      { "upload:    ", &ul_time_ms },
      { "calc:      ", &calc_time },
      { "download:  ", &dl_time },
      { "visualize: ", &visualize_time_ms },
      { "total:     ", &total_time_ms },
    };

    for (auto t : times) {
      std::stringstream ss;
      ss << t.text << *t.time << "ms";
      cv::putText(result, ss.str(), pos, font, scale, color);
      pos.y += 15;
    }
  }

  mVisualizationImage->update(result);
}

void OpticalFlow::toggle_visualization()
{
  if (mVisualization == OpticalFlow::OPTICAL_FLOW_VISUALIZATION_ARROWS) {
    mVisualization = OpticalFlow::OPTICAL_FLOW_VISUALIZATION_BLOCKS;
  } else {
    mVisualization = OpticalFlow::OPTICAL_FLOW_VISUALIZATION_ARROWS;
  };
}
