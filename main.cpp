#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "movingObject.h"

using namespace std;

void search_moving_objects(cv::Mat &threshold, cv::Mat &liveFeed)
{
  int const max_objects = 50;
  double const min_object_area = 20 * 20;
  vector<vector<cv::Point>> contours;
  vector<cv::Vec4i> hierarchy;
  vector<MovingObject> objects;

  cv::findContours(threshold, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

  if ((contours.size() < max_objects) && !contours.empty()) {

    // hierarchy vector gives the hierarchy of found contours. hierarchy[i][0] is the next contour
    // in hierarchy
    for (int i = 0; i >= 0; i = hierarchy[i][0]) {

      cv::Moments moment = moments(contours[i]);
      if (moment.m00 > min_object_area) {

        MovingObject obj(contours[i]);

        int len = objects.size();
        for (int i = 0; i < len; i++) {
          if (obj.inRange(objects[i])) {
            obj.setColor(cv::Scalar(0, 255, 0));
            objects.emplace_back(obj, objects[i]);
          }
        }

        objects.emplace_back(std::move(obj));
      }
    }

  } else if (contours.empty()) {
    cv::putText(liveFeed, "no movement detected",
                cv::Point(50, 50), 
                0, 0.5, cv::Scalar(255, 255, 255));
  } else {
    cv::putText(liveFeed, "too much noise in video stream",
                cv::Point(50, 50), 
                0, 0.5, cv::Scalar(255, 255, 255));
  }

  for_each(objects.begin(), objects.end(), [&liveFeed](MovingObject const &o)
           {
             o.draw(liveFeed);
           });

}

void on_trackbar( int, void* ) { }

void refine_diff_image(cv::Mat const &diff, cv::Mat &threshold)
{
  cv::Mat thresh;
  static int threshold_sensitivity = 20;
  static int blur_size = 50;
  static int threshold_sensitivity2 = 30;
 
  {
    cv::namedWindow("refine_diff_image", 0);
    cv::createTrackbar("Threshold sensitivity", "refine_diff_image", &threshold_sensitivity, 255, on_trackbar);
    cv::createTrackbar("Blur size", "refine_diff_image", &blur_size, 255, on_trackbar);
    cv::createTrackbar("Threshold sensitivity2", "refine_diff_image", &threshold_sensitivity2, 255, on_trackbar);
  }

  cv::threshold(diff, thresh, threshold_sensitivity, 255, cv::THRESH_BINARY);
  cv::blur(thresh, threshold, cv::Size(blur_size, blur_size));
  cv::threshold(threshold, threshold, threshold_sensitivity2, 255, cv::THRESH_BINARY);

  {
    cv::imshow("Differential image", diff);
    cv::imshow("Initial Threshold image", thresh);
    cv::imshow("Final Threshold image", threshold);
  }
}

void capture_loop(cv::VideoCapture &camera)
{
  bool exit = false;
  cv::Mat image;
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  // read first image, before entering loop
  camera.read(image);
  cv::cvtColor(image, nowGrayscale, cv::COLOR_BGR2GRAY);

  while (!exit) {
    nowGrayscale.copyTo(lastGrayscale);

    // take new image and convert to grayscale
    camera.read(image);
    cv::cvtColor(image, nowGrayscale, cv::COLOR_BGR2GRAY);

    // difference between the two images, will filter static areas
    cv::absdiff(lastGrayscale, nowGrayscale, diff);

    refine_diff_image(diff, thresh);

    search_moving_objects(thresh, image);

    cv::imshow("Live Feed", image);

    // check for button press for 10ms. necessary for opencv to refresh windows
    char key = cv::waitKey(10);
    switch (key) {
      case 'q':
        exit = true;
        break;
      case -1:
      default:
        break;
    }
  }
}

int main(int argc, char **argv)
{
  int cam = 0;
  if (argc > 1) {
    cerr << "camera number not set, assuming 0" << endl;
    cam = atoi(argv[1]);
  }

  cv::VideoCapture camera;
  camera.open(cam);
  if (!camera.isOpened()) {
    cerr << "Error opening camera " << cam << endl;
    return -1;
  }

  capture_loop(camera);

  camera.release();
  return 0;
}
