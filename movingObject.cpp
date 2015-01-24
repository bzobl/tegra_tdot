#include "movingObject.h"

MovingObject::MovingObject(std::vector<cv::Point> contours) : mContours(contours)
{
  mBoundingBox = cv::boundingRect(contours);
}

std::vector<cv::Point> MovingObject::getContours()
{
  return mContours;
}

cv::Point MovingObject::getCenter()
{
  return cv::Point(mBoundingBox.x + mBoundingBox.width/2, mBoundingBox.y + mBoundingBox.height/2);
}

cv::Rect MovingObject::getBoundingBox()
{
  return mBoundingBox;
}
