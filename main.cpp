#include <chrono>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/gpu/gpu.hpp"

using namespace std;
using namespace cv;
using namespace cv::gpu;

template <typename T>
inline T mapVal(T x, T a, T b, T c, T d)
{
    x = ::max(::min(x, b), a);
    return c + (d-c) * (x-a) / (b-a);
}

static void colorizeFlow(const Mat &u, const Mat &v, Mat &dst)
{
    double uMin, uMax;
    minMaxLoc(u, &uMin, &uMax, 0, 0);
    double vMin, vMax;
    minMaxLoc(v, &vMin, &vMax, 0, 0);
    uMin = ::abs(uMin); uMax = ::abs(uMax);
    vMin = ::abs(vMin); vMax = ::abs(vMax);
    float dMax = static_cast<float>(::max(::max(uMin, uMax), ::max(vMin, vMax)));

    dst.create(u.size(), CV_8UC3);
    for (int y = 0; y < u.rows; ++y)
    {
        for (int x = 0; x < u.cols; ++x)
        {
            dst.at<uchar>(y,3*x) = 0;
            dst.at<uchar>(y,3*x+1) = (uchar)mapVal(-v.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
            dst.at<uchar>(y,3*x+2) = (uchar)mapVal(u.at<float>(y,x), -dMax, dMax, 0.f, 255.f);
        }
    }
}

void capture_loop(cv::VideoCapture &camera)
{
  using time = std::chrono::time_point<std::chrono::high_resolution_clock>;
  time upload_start, upload_stop,
       calc_start, calc_stop,
       download_start, download_stop,
       show_start, show_stop;

  bool exit = false;
  cv::Mat image, grayscale;
  cv::gpu::GpuMat gImg1, gImg2;
  cv::gpu::GpuMat *nowGImg = &gImg1;
  cv::gpu::GpuMat *lastGImg = &gImg2;

  FarnebackOpticalFlow flow;

  GpuMat d_flowx, d_flowy;
  Mat flowxy, flowx, flowy, result;

  // read first image, before entering loop
  camera.read(image);
  cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  nowGImg->upload(grayscale);

  while (!exit) {
    // swap pointers to avoid reallocating memory on gpu
    ::swap(nowGImg, lastGImg);

    // take new image and convert to grayscale
    upload_start = std::chrono::high_resolution_clock::now();
    camera.read(image);
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    nowGImg->upload(grayscale);
    upload_stop = std::chrono::high_resolution_clock::now();

    calc_start = std::chrono::high_resolution_clock::now();
    flow(*lastGImg, *nowGImg, d_flowx, d_flowy);
    calc_stop = std::chrono::high_resolution_clock::now();

    download_start = std::chrono::high_resolution_clock::now();
    d_flowx.download(flowx);
    d_flowy.download(flowy);
    download_stop = std::chrono::high_resolution_clock::now();

    show_start = std::chrono::high_resolution_clock::now();
    colorizeFlow(flowx, flowy, result);
    cv::imshow("Live Feed", result);
    show_stop = std::chrono::high_resolution_clock::now();

    // print times
    std::cout << "Times (in ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(upload_stop - upload_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(calc_stop - calc_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(download_stop - download_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(show_stop - show_start).count()
              << std::endl;

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

  int gpu = 0;
  cv::gpu::setDevice(gpu);
  cv::gpu::resetDevice();
  cv::gpu::DeviceInfo info;
  std::cout << "using GPU" << gpu << ": "
            << info.freeMemory() / 1024 / 1024 << " / "
            << info.totalMemory() / 1024 / 1024 << " MB in use" << std::endl;

  capture_loop(camera);

  camera.release();
  return 0;
}
