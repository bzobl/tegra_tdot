#ifndef MOVING_OBJECT_H_INCLUDED
#define MOVING_OBJECT_H_INCLUDED

#include <opencv/cv.h>

class MovingObject {

private:
  std::vector<cv::Point> mContours;
  cv::Rect mBoundingBox;

public:

  MovingObject(std::vector<cv::Point> contours);

  std::vector<cv::Point> getContours();
  cv::Point getCenter();
  cv::Rect getBoundingBox();

};

#endif
