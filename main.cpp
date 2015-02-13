#include <iostream>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace std;
using namespace cv;

vector<cv::Rect> run_facerecognition(cv::Mat &live_image, cv::CascadeClassifier &cascade)
{
  vector<cv::Rect> faces;

  cv::Mat gray;
  cv::cvtColor(live_image, gray, CV_BGR2GRAY);

  //cascade.detectMultiScale(live_image, faces, 1.15, 3, CASCADE_SCALE_IMAGE, Size(30,30));
  cascade.detectMultiScale(gray, faces);

  for (Rect face : faces) {
    rectangle(live_image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
              Scalar(255, 255, 255), 1, 4);
  }
  return faces;
}

void insert_alpha_image(cv::Mat &result, cv::Rect roi, cv::Mat image)
{
  cv::Mat imageBGR(image.rows, image.cols, CV_8UC3);
  cv::Mat imageALPHA(image.rows, image.cols, CV_8UC1);
  Mat out[] = {imageBGR, imageALPHA};
  int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};

  cv::mixChannels(&image, 1, out, 2, from_to, 4);

  for (int i = 0; i < roi.width; i++) {
    for (int j = 0; j < roi.height; j++) {
      if (imageALPHA.at<uchar>(j, i) > 0) {
        result.at<Vec3b>(roi.y + j, roi.x + i) = imageBGR.at<Vec3b>(j, i);
      }
    }
  }
}

void capture_loop(cv::VideoCapture &camera)
{
  bool exit = false;
  cv::Mat image;
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  //string const face_xml = "face.xml";
  string const face_xml = "./haarcascade_frontalface_default.xml";
  cv::CascadeClassifier face_cascade;
  face_cascade.load(face_xml);

  vector<cv::Mat> hats;
  hats.push_back(cv::imread("sombrero.png", CV_LOAD_IMAGE_UNCHANGED));

  while (!exit) {
    // take new image
    camera.read(image);

    vector<cv::Rect> faces = run_facerecognition(image, face_cascade);

    for (Rect face : faces) {
      cv::Mat hat = hats[0];
      //scale hat
      double ratio = hat.cols / hat.rows;
      cv::resize(hat, hat, Size(face.width * 2, face.width * 2 / ratio), 1.0, 1.0, INTER_CUBIC);

      if (   (face.y - hat.rows) > 0
          && (face.x - hat.cols) > 0) {
        insert_alpha_image(image, cv::Rect(face.x - hat.cols/4, face.y - hat.rows, hat.cols, hat.rows), hat);
      }

      //hat.copyTo(result(cv::Rect(face.x, face.y, hat.cols, hat.rows)));
    }

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
