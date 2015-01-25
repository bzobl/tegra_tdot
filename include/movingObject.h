#ifndef MOVING_OBJECT_H_INCLUDED
#define MOVING_OBJECT_H_INCLUDED

#include <opencv/cv.h>

class MovingObject {

private:
  std::vector<cv::Point> mContours;
  cv::Rect mBoundingBox;
  cv::Scalar mBBColor = cv::Scalar(255, 255, 255);

  static int const MINIMAL_DISTANCE = 30;

public:
  MovingObject(std::vector<cv::Point> &contours);
  MovingObject(MovingObject &o, MovingObject & o2);

  std::vector<cv::Point> getContours() const;
  cv::Point getCenter() const;
  cv::Rect getBoundingBox() const;

  void setColor(cv::Scalar color);

  bool inRange(MovingObject const &other) const;
  void draw(cv::Mat &image) const;
};

#endif
