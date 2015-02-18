#include <iostream>
#include <vector>
#include <thread>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/gpu/gpu.hpp"

#include "livestream.h"

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

void facerecognition_thread(LiveStream &stream, cv::gpu::CascadeClassifier_GPU &cascade,
                            AlphaImage &hat, bool &exit)
{
  cv::Mat frame;

  while (!exit) {
    stream.getFrame(frame);

    vector<cv::Rect> faces = run_facerecognition_gpu(frame, cascade);

    std::unique_lock<std::mutex> l(stream.getOverlayMutex());
    stream.resetOverlay();
    for (Rect face : faces) {
      stream.writeOverlayImage(hat, face.width * 2,
                               face.x - face.width/2, face.y - hat.height(face.width));
    }
  }
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

  std::thread detection_thread(facerecognition_thread,
                               std::ref(stream), std::ref(face_cascade_gpu),
                               std::ref(hats[0]), std::ref(exit));

  const std::string live_feed_window = "Live Feed";
  /*
  cv::namedWindow(live_feed_window, CV_WINDOW_NORMAL);
  cv::setWindowProperty(live_feed_window, CV_WND_PROP_FULLSCREEN, CV_WND_PROP_FULLSCREEN);
  cv::moveWindow(live_feed_window, 0, 0);
  cv::resizeWindow(live_feed_window, 1920, 1080);
  */

  while (!exit) {
    double t = (double) getTickCount();

    // take new image
    stream.nextFrame(image);

    // TODO move to separate thread
    //facerecognition_thread(stream, face_cascade_gpu, hats[0]);

    stream.applyOverlay(image);

    t = ((double) getTickCount() - t) / getTickFrequency();
    std::stringstream ss;
    ss << "Time: " << t*1000 << "ms | FPS: " << 1/t;

    cv::putText(image, ss.str(), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));

    cv::imshow(live_feed_window, image);

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

  detection_thread.join();
}

int main(int argc, char **argv)
{
  int cam = 0;
  if (argc > 1) {
    cam = atoi(argv[1]);
  }
  
  LiveStream live(cam);
  //LiveStream live(cam, 1920, 1080);
  //LiveStream live(cam, 640, 480);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << cam << endl;
    return -1;
  }

  capture_loop(live);

  return 0;
}
