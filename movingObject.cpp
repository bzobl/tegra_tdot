#include "movingObject.h"

#include <cmath>

MovingObject::MovingObject(std::vector<cv::Point> &contours) : mContours(contours)
{
  mBoundingBox = cv::boundingRect(contours);
}

MovingObject::MovingObject(MovingObject &o, MovingObject & o2)
{
  mContours = o.mContours;
  mContours.insert(mContours.begin(), o2.mContours.begin(), o2.mContours.end());
  mBoundingBox = cv::boundingRect(mContours);
}

std::vector<cv::Point> MovingObject::getContours() const
{
  return mContours;
}

cv::Point MovingObject::getCenter() const
{
  return cv::Point(mBoundingBox.x + mBoundingBox.width/2, mBoundingBox.y + mBoundingBox.height/2);
}

cv::Rect MovingObject::getBoundingBox() const
{
  return mBoundingBox;
}

void MovingObject::setColor(cv::Scalar color)
{
  mBBColor = color;
}

bool MovingObject::inRange(MovingObject const &other) const
{
  double delta_x = std::abs(getCenter().x - other.getCenter().x);
  double delta_y = std::abs(getCenter().y - other.getCenter().y);
  double center_distance = std::sqrt(delta_x * delta_x + delta_y * delta_y);

  if ((center_distance - other.mBoundingBox.width/2 - mBoundingBox.width/2) < MINIMAL_DISTANCE) {
    //TODO use width and height of bounding boxes
    return true;
  }
  return false;
}

void MovingObject::draw(cv::Mat &image) const
{
  cv::rectangle(image, mBoundingBox, mBBColor);
  std::stringstream ss;
  ss << "(" << getCenter().x << "," << getCenter().y << ")";
  cv::putText(image, ss.str(),
              cv::Point(mBoundingBox.x, mBoundingBox.y + mBoundingBox.height + 10), 0, 0.5, mBBColor);
}
