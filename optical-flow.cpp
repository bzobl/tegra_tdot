#include "optical-flow.h"

#include <algorithm>
#include <cassert>

OpticalFlow::OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization)
                        : mStream(stream), mVisualizationImage(visualization)
{
  mNowGpuImg = &mGpuImg1;
  mLastGpuImg = &mGpuImg2;

  load_new_frame();
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

void OpticalFlow::visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy,
                                            cv::Mat &result)
{
  int const width = flowx.cols;
  int const height = flowx.rows;
  double l_max = 10;

  for (int y = 0; y < height; y += 10) {
    for (int x = 0; x < width; x += 10) {
      double dx = flowx.at<float>(y, x);
      double dy = flowy.at<float>(y, x);

      cv::Point p(x, y);
      double l = std::max(std::sqrt(dx*dx + dy*dy), l_max);

      if (l > 0) {
        cv::Point p2(p.x + (int)dx, p.y + (int)dy);
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
}

void OpticalFlow::operator()()
{
  assert(isReady());
  cv::Mat flowx, flowy;
  cv::Mat result(mStream.height(), mStream.width(), CV_8UC3);

  double ul_start = (double) cv::getTickCount();
  load_new_frame();
  double ul_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  double calc_time, dl_time;
  use_farneback(flowx, flowy, calc_time, dl_time);

  double visualize_start = (double) cv::getTickCount();
  visualize_optical_flow(flowx, flowy, result);
  double visualize_time_ms = ((double) cv::getTickCount() - visualize_start) / cv::getTickFrequency() * 1000;

  double total_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  std::stringstream ss;
  cv::Point pos(50, 50);
  cv::fontFace font = cv::FONT_HERSHEY_PLAIN;
  cv::Scalar color(255, 255, 255);
  double scale = 0.5;

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
    ss << t.text << *t.time << "ms";
    cv::putText(result, ss.str(), pos, font, scale, color);
    pos.y += 10;
  }

  mVisualizationImage.update(result);
}
