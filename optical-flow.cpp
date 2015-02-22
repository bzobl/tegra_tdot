#include "optical-flow.h"

#include <algorithm>
#include <cassert>

OpticalFlow::OpticalFlow(LiveStream &stream, ThreadSafeMat &visualization)
                        : mStream(stream), mVisualizationImage(visualization)
{
  mNowGpuImg = &mGpuImg1;
  mLastGpuImg = &mGpuImg2;
}

bool OpticalFlow::isReady()
{
  return mStream.isOpened();
}

void OpticalFlow::use_farneback(cv::Mat &flowx, cv::Mat &flowy,
                                double &calc_time_ms, double &dl_time_ms)
{
  cv::cuda::GpuMat d_flowx, d_flowy;
  cv::Mat flowx, flowy;

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
  cv::Mat image, flowx, flowy;
  cv::Mat result(mStream.height(), mStream.width(), CV_8UC3);

  double ul_start = (double) cv::getTickCount();
  mStream.getFrame(image);
  cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
  mNowGpuImg->upload(image);
  double ul_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  double calc_time, dl_time;
  use_farneback(flowx, flowy, calc_time, dl_time);

  double visualize_start = (double) cv::getTickCount();
  visualize_optical_flow(flowx, flowy, result);
  double visualize_time_ms = ((double) cv::getTickCount() - visualize_start) / cv::getTickFrequency() * 1000;

  // swap pointers to avoid reallocating memory on gpu
  std::swap(mNowGpuImg, mLastGpuImg);

  double total_time_ms = ((double) cv::getTickCount() - ul_start) / cv::getTickFrequency() * 1000;

  std::stringstream ss;
  ss << "upload: " << ul_time_ms << "ms" << std::endl
     << "calc:   " << calc_time << "ms" << std::endl
     << "download: " << dl_time << "ms" << std::endl
     << "visualize: " << visualize_time_ms << "ms" << std::endl
     << "total: " << total_time_ms << "ms | FPS: " << 1000/total_time_ms << std::endl;

  cv::putText(result, ss.str(), cv::Point(50, 50), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(255, 255, 255));
  //cv::imshow("OptFlow", result);
  mVisualizationImage.update(result);
}
