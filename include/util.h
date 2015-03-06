#ifndef UTIL_H_INCLUDED
#define UTIL_H_INCLUDED

#include <string>
#include <sstream>
#include <vector>

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

struct PrintableTime {
  std::string text;
  double *time;
};

inline cv::Point print_times(cv::Mat image, cv::Point start, std::vector<PrintableTime> times, bool is_ms = false)
{
  for (auto t : times) {
    std::stringstream ss;
    double time = *t.time;
    if (!is_ms) {
      time *= 1000.0;
    }
    ss << t.text << time << "ms";

    cv::putText(image, ss.str(), start, cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(255, 255, 255));
    start.y += 15;
  }

  return start;
}

#endif
