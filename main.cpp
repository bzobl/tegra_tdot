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

using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

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

cv::Mat &&use_farneback(cv::gpu::GpuMat *last, cv::gpu::GpuMat *now,
                        timepoint &calc_start, timepoint &calc_stop, 
                        timepoint &download_start, timepoint &download_stop)
{
  FarnebackOpticalFlow flow;

  GpuMat d_flowx, d_flowy;
  Mat flowxy, flowx, flowy, result;

  calc_start = std::chrono::high_resolution_clock::now();
  flow(*last, *now, d_flowx, d_flowy);
  calc_stop = std::chrono::high_resolution_clock::now();

  download_start = std::chrono::high_resolution_clock::now();
  d_flowx.download(flowx);
  d_flowy.download(flowy);
  download_stop = std::chrono::high_resolution_clock::now();

  colorizeFlow(flowx, flowy, result);
}

void getFlowField(const Mat& u, const Mat& v, Mat& flowField)
{
    float maxDisplacement = 1.0f;

    for (int i = 0; i < u.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);

        for (int j = 0; j < u.cols; ++j)
        {
            float d = max(fabsf(ptr_u[j]), fabsf(ptr_v[j]));

            if (d > maxDisplacement)
                maxDisplacement = d;
        }
    }

    flowField.create(u.size(), CV_8UC4);

    for (int i = 0; i < flowField.rows; ++i)
    {
        const float* ptr_u = u.ptr<float>(i);
        const float* ptr_v = v.ptr<float>(i);


        Vec4b* row = flowField.ptr<Vec4b>(i);

        for (int j = 0; j < flowField.cols; ++j)
        {
            row[j][0] = 0;
            row[j][1] = static_cast<unsigned char> (mapVal (-ptr_v[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][2] = static_cast<unsigned char> (mapVal ( ptr_u[j], -maxDisplacement, maxDisplacement, 0.0f, 255.0f));
            row[j][3] = 255;
        }
    }
}

cv::Mat use_brox(cv::gpu::GpuMat *last, cv::gpu::GpuMat *now,
                 timepoint &calc_start, timepoint &calc_stop, 
                 timepoint &download_start, timepoint &download_stop)
{
  float alpha = 0.197;
  float scale = 0.8;
  float gamma = 50.0;
  int inner_iterations = 10;
  int outer_iterations = 77;
  int solver_iterations = 10;

  GpuMat d_fu, d_fv;

  BroxOpticalFlow d_flow(alpha, gamma, scale, inner_iterations, outer_iterations, solver_iterations);

  d_flow(*last, *now, d_fu, d_fv);

  Mat flowField;
  getFlowField(Mat(d_fu), Mat(d_fv), flowField);
  return flowField;
}

void capture_loop(cv::VideoCapture &camera)
{
  timepoint upload_start, upload_stop,
            calc_start, calc_stop,
            download_start, download_stop,
            total_start, total_stop;

  bool exit = false;
  bool farneback = false;
  cv::Mat image, grayscale, result;

  cv::gpu::GpuMat gImg1, gImg2;
  cv::gpu::GpuMat *nowGImg = &gImg1;
  cv::gpu::GpuMat *lastGImg = &gImg2;

  // read first image, before entering loop
  camera.read(image);
  cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
  nowGImg->upload(grayscale);

  while (!exit) {
    total_start = std::chrono::high_resolution_clock::now();
    // swap pointers to avoid reallocating memory on gpu
    ::swap(nowGImg, lastGImg);

    // take new image and convert to grayscale
    upload_start = std::chrono::high_resolution_clock::now();
    camera.read(image);
    cv::cvtColor(image, grayscale, cv::COLOR_BGR2GRAY);
    nowGImg->upload(grayscale);
    upload_stop = std::chrono::high_resolution_clock::now();

    if (farneback) {
      result = use_farneback(lastGImg, nowGImg, calc_start, calc_stop, download_start, download_stop);
    } else {
      result = use_brox(lastGImg, nowGImg, calc_start, calc_stop, download_start, download_stop);
    }

    cv::imshow("Live Feed", result);

    total_stop = std::chrono::high_resolution_clock::now();
    // print times
    std::cout << "Times (in ms): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(upload_stop - upload_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(calc_stop - calc_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(download_stop - download_start).count()
              << " | "
              << std::chrono::duration_cast<std::chrono::milliseconds>(total_stop - total_start).count()
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
