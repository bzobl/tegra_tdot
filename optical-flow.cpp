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
  mFarneback.fastPyramids = true;
  mFarneback.winSize = 13;          // averaging window size
  mFarneback.numIters = 10;         // iterations per pyramid level
  mFarneback.polyN = 5;             // size of pixel neighborhood. usally 5 or 7
  mFarneback.polySigma = 1.1;       // standard deviation for gaussian usually 1.1 or 1.5
  mFarneback.flags = 0;
}

bool OpticalFlow::isReady()
{
  return mStream.isOpened();
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

cv::Mat OpticalFlow::visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy)
{
  int const width = flowx.cols;
  int const height = flowx.rows;
  double const l_threshold = 0.2;
  cv::Mat result(flowx.rows, flowy.cols, CV_8UC3);;

  for (int y = 0; y < height; y += 10) {
    for (int x = 0; x < width; x += 10) {
      double dx = flowx.at<float>(y, x);
      double dy = flowy.at<float>(y, x);

      double l = std::sqrt(dx*dx + dy*dy);

      if ((l > l_threshold)) {
        cv::Point p(x, y);
        cv::Point p2(x + dx, y + dy);
        cv::Scalar color;
        if (p.y > p2.y) {
          color = cv::Scalar(128, 128, 0);
        } else {
          color = cv::Scalar(0, 0, 255);
        }
        cv::arrowedLine(result, p, p2, color);
      }
    }
  }
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
  result = visualize_optical_flow(flowx, flowy);
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
      pos.y += 10;
    }
  }

  mVisualizationImage->update(result);
}
