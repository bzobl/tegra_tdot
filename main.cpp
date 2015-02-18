#include <iostream>
#include <vector>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "livestream.h"

using namespace std;
using namespace cv;

class AlphaImage {

private:
  cv::Mat color;
  cv::Mat alpha;

  double ratio;

public:
  AlphaImage(string filename)
  {
    cv::Mat image = cv::imread(filename, CV_LOAD_IMAGE_UNCHANGED);
    color = cv::Mat(image.rows, image.cols, CV_8UC3);
    alpha = cv::Mat(image.rows, image.cols, CV_8UC1);
    Mat out[] = {color, alpha};
    int from_to[] = {0, 0, 1, 1, 2, 2, 3, 3};
    cv::mixChannels(&image, 1, out, 2, from_to, 4);

    ratio = image.cols / image.rows;
  }

  int width() const { return color.cols; }
  int height() const { return color.rows; }

  int height(int width) const { return color.rows / ratio; }

  void write_to_image(cv::Mat &image, int width, int x, int y)
  {
    //scale image
    cv::Mat c, a;
    cv::resize(color, c, Size(width, width / ratio), 1.0, 1.0, INTER_CUBIC);
    cv::resize(alpha, a, Size(width, width / ratio), 1.0, 1.0, INTER_CUBIC);

    for (int i = 0; i < c.cols; i++) {
      for (int j = 0; j < c.rows; j++) {
        if (a.at<uchar>(j, i) > 0) {
          if (((y + j) > 0) && ((x + i) > 0)) {
            image.at<Vec3b>(y + j, x + i) = c.at<Vec3b>(j, i);
          }
        }
      }
    }
  }
};

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

vector<cv::Rect> run_facerecognition_gpu(cv::Mat &live_image, cv::gpu::CascadeClassifier_GPU &cascade)
{
  vector<cv::Rect> faces;
  cv::gpu::GpuMat d_faces;
  cv::Mat h_faces;

  cv::Mat gray;
  cv::cvtColor(live_image, gray, CV_BGR2GRAY);
  cv::gpu::GpuMat d_gray(gray);

  int n_detected = cascade.detectMultiScale(d_gray, d_faces, 1.2, 8, Size(40, 40));
  
  d_faces.colRange(0, n_detected).download(h_faces);
  Rect *prect = h_faces.ptr<Rect>();

  for (int i = 0; i < n_detected; i++) {
    faces.push_back(prect[i]);
  }

  return faces;
}

void capture_loop(LiveStream &stream)
{
  bool exit = false;
  cv::Mat image;
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  //string const face_xml = "./face.xml";
  string const face_xml = "../opencv/data/haarcascades/haarcascade_frontalface_alt.xml";

  /*
  cv::CascadeClassifier face_cascade;
  if (!face_cascade.load(face_xml)) {
    cout << "Could not load " << face_xml << std::endl;
    return;
  }
  */

  cv::gpu::CascadeClassifier_GPU face_cascade_gpu;
  if (!face_cascade_gpu.load(face_xml)) {
    cout << "GPU Could not load " << face_xml << std::endl;
    return;
  }

  std::cout << "all loaded" << std::endl;

  vector<AlphaImage> hats;
  hats.emplace_back("sombrero.png");

  while (!exit) {
    double t = (double) getTickCount();

    cout << "Loading new image" << endl;

    // take new image
    stream.nextFrame(image);

    //vector<cv::Rect> faces = run_facerecognition(image, face_cascade);
    vector<cv::Rect> faces = run_facerecognition_gpu(image, face_cascade_gpu);

    /*
    for (Rect face : faces) {
      AlphaImage *hat = &hats[0];

      hat->write_to_image(image, face.width * 2, 
                          face.x - face.width/2, face.y - hat->height(face.width));
    }
    */

    stream.applyOverlay(image);

    t = ((double) getTickCount() - t) / getTickFrequency();
    std::stringstream ss;
    ss << "Time: " << t*1000 << "ms | FPS: " << 1/t;

    cv::putText(image, ss.str(), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(0, 0, 0));

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

  
  LiveStream live(cam);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << cam << endl;
    return -1;
  }

  capture_loop(live);

  return 0;
}
