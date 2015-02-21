#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"

#include "livestream.h"

using namespace std;
using namespace cv;

using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

cv::Mat visualize_optical_flow(cv::Mat const &flowx, cv::Mat const &flowy)
{
  int const width = flowx.cols;
  int const height = flowx.rows;
  cv::Mat arrows = cv::Mat(height, width, CV_8UC3);
  double l_max = 10;

  for (int y = 0; y < height; y += 10) {
    for (int x = 0; x < width; x += 10) {
      double dx = flowx.at<float>(y, x);
      double dy = flowy.at<float>(y, x);

      cv::Point p(x, y);
      double l = std::max(std::sqrt(dx*dx + dy*dy), l_max);

      if (l > 0) {
        //double spin_size = 5.0 * l/l_max;
        cv::Point p2(p.x + (int)dx, p.y + (int)dy);
        cv::arrowedLine(arrows, p, p2, cv::Scalar(128, 128, 0));

      }
    }
  }

  return arrows;
}

cv::Mat optical_flow_farneback(cv::cuda::GpuMat *last, cv::cuda::GpuMat *now,
                               timepoint &calc_start, timepoint &calc_stop,
                               timepoint &download_start, timepoint &download_stop)
{
  cv::cuda::FarnebackOpticalFlow flow;

  cv::cuda::GpuMat d_flowx, d_flowy;
  cv::Mat flowxy, flowx, flowy, result;

  calc_start = std::chrono::high_resolution_clock::now();
  flow(*last, *now, d_flowx, d_flowy);
  calc_stop = std::chrono::high_resolution_clock::now();

  download_start = std::chrono::high_resolution_clock::now();
  d_flowx.download(flowx);
  d_flowy.download(flowy);
  download_stop = std::chrono::high_resolution_clock::now();

  return visualize_optical_flow(flowx, flowy);
}

void optical_flow_thread(LiveStream &stream, std::atomic<bool> &exit)
{
  timepoint upload_start, upload_stop,
            calc_start, calc_stop,
            download_start, download_stop,
            total_start, total_stop;

  cv::Mat image, grayscale, result;

  cv::cuda::GpuMat gImg1, gImg2;
  cv::cuda::GpuMat *nowGImg = &gImg1;
  cv::cuda::GpuMat *lastGImg = &gImg2;

  // read first image, before entering loop
  stream.getFrame(image);
  cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  nowGImg->upload(grayscale);

  while (!exit) {
    total_start = std::chrono::high_resolution_clock::now();

    // swap pointers to avoid reallocating memory on gpu
    ::swap(nowGImg, lastGImg);

    // take new image and convert to grayscale
    upload_start = std::chrono::high_resolution_clock::now();

    stream.nextFrame(image);
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    nowGImg->upload(grayscale);

    upload_stop = std::chrono::high_resolution_clock::now();

    result = optical_flow_farneback(lastGImg, nowGImg, calc_start, calc_stop, download_start, download_stop);

    total_stop = std::chrono::high_resolution_clock::now();
    // print times
    std::stringstream ss;
    ss << "Times (in ms): "
       << std::chrono::duration_cast<std::chrono::milliseconds>(upload_stop - upload_start).count()
       << " | "
       << std::chrono::duration_cast<std::chrono::milliseconds>(calc_stop - calc_start).count()
       << " | "
       << std::chrono::duration_cast<std::chrono::milliseconds>(download_stop - download_start).count()
       << " | "
       << std::chrono::duration_cast<std::chrono::milliseconds>(total_stop - total_start).count()
       << std::endl;

    cv::putText(image, ss.str(), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
    cv::imshow("OptFlow", result);

    // check for button press for 10ms. necessary for opencv to refresh windows
    char key = cv::waitKey(10);
    switch (key) {
      case 'q':
        exit = true;
        break;
      default:
        break;
    }
  }
}

vector<cv::Rect> run_facerecognition(cv::Mat &live_image, cv::CascadeClassifier &cascade)
{
  vector<cv::Rect> faces;

  cv::Mat gray;
  cv::cvtColor(live_image, gray, COLOR_BGR2GRAY);

  //cascade.detectMultiScale(live_image, faces, 1.15, 3, CASCADE_SCALE_IMAGE, Size(30,30));
  cascade.detectMultiScale(gray, faces);

  for (Rect face : faces) {
    rectangle(live_image, Point(face.x, face.y), Point(face.x+face.width, face.y+face.height),
              Scalar(255, 255, 255), 1, 4);
  }
  return faces;
}

vector<cv::Rect> run_facerecognition_gpu(cv::Mat &live_image, cv::cuda::CascadeClassifier_CUDA &cascade)
{
  vector<cv::Rect> faces;
  cv::cuda::GpuMat d_faces;
  cv::Mat h_faces;

  cv::Mat gray;
  cv::cvtColor(live_image, gray, COLOR_BGR2GRAY);
  cv::cuda::GpuMat d_gray(gray);

  int n_detected = cascade.detectMultiScale(d_gray, d_faces, 1.2, 8, Size(40, 40));

  d_faces.colRange(0, n_detected).download(h_faces);
  Rect *prect = h_faces.ptr<Rect>();

  for (int i = 0; i < n_detected; i++) {
    faces.push_back(prect[i]);
  }

  return faces;
}

void facerecognition_thread(LiveStream &stream, std::string const & cascade_face,
                            AlphaImage &hat, std::atomic<bool> &exit)
{
  cv::Mat frame;

  cv::cuda::CascadeClassifier_CUDA cascade(cascade_face);
  if (cascade.empty()) {
    cout << "GPU Could not load " << cascade_face << std::endl;
    return;
  }

  std::cout << "cascade loaded" << std::endl;

  while (!exit) {
    stream.getFrame(frame);

    vector<cv::Rect> faces = run_facerecognition_gpu(frame, cascade);

    // for the duration of resetting the overlay no other thread must use the overlay
    {
      std::unique_lock<std::mutex> l(stream.getOverlayMutex());
      stream.resetOverlay();
      for (Rect face : faces) {
        int width = face.width * 2;
        int height = hat.height(width);
        int x = face.x - width/4;
        int y = face.y - height;
        stream.addImageToOverlay(hat, width, x, y);
      }
    }
  }
}

void capture_loop(LiveStream &stream)
{
  std::atomic<bool> exit(false);
  cv::Mat image;
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  string const face_xml = "./face.xml";
  //string const face_xml = "../opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";

  vector<AlphaImage> hats;
  hats.emplace_back("sombrero.png");

  std::thread opt_flow_thread(optical_flow_thread, std::ref(stream), std::ref(exit));
  opt_flow_thread.join();

  std::thread detection_thread(facerecognition_thread,
                               std::ref(stream), face_xml,
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

  //LiveStream live(cam);
  //LiveStream live(cam, 1920, 1080);
  //LiveStream live(cam, 1280, 720);
  LiveStream live(cam, 640, 480);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << cam << endl;
    return -1;
  }

  capture_loop(live);

  return 0;
}
