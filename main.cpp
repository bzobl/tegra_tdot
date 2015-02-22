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
#include "thread-safe-mat.h"

using namespace std;
using namespace cv;

using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct Options {
  int cam_num = 0;
  int width = 640;
  int height = 480;
  bool face_detect = false;
  bool optical_flow = false;
};

void usage(std::string progname)
{
  std::cout << "usage:" << std::endl
            << progname << " [OPTIONS]" << std::endl
            << std::endl
            << "Options:" << std::endl
            << " -c, --camera: Number of the camera to capture. E.g. 0 for /dev/video0" << std::endl
            << " -w, --width: Width of the captured image" << std::endl
            << " -h, --height: Height of the captured image" << std::endl
            << " -f, --face-detect: Enable face detection and augmented reality" << std::endl
            << " -o, --optical-flow: Enable optical flow analysis" << std::endl
            << " -h, --help: Show this help" << std::endl
            << std::endl;
}

std::ostream &operator<<(ostream &out, Options const &o)
{
  out << "Camera:       " << o.cam_num << std::endl
      << "Width:        " << o.width << std::endl
      << "Height:       " << o.height << std::endl
      << "Facedetect:   " << std::boolalpha << o.face_detect << std::endl
      << "Optical Flow: " << std::boolalpha << o.optical_flow << std::endl;
  return out;
}

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
        cv::Scalar color;
        if (p.y > p2.y) {
          color = cv::Scalar(128, 128, 0);
        } else {
          color = cv::Scalar(0, 0, 255);
        }
        cv::arrowedLine(arrows, p, p2, color);

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

void optical_flow_thread(LiveStream &stream, ThreadSafeMat &visualization, std::atomic<bool> &exit)
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

    cv::putText(result, ss.str(), Point(50, 50), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));
    //cv::imshow("OptFlow", result);
    visualization.update(result);

    // check for button press for 10ms. necessary for opencv to refresh windows
    /*
    char key = cv::waitKey(10);
    switch (key) {
      case 'q':
        exit = true;
        break;
      default:
        break;
    }
    */
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

void capture_loop(LiveStream &stream, Options const &opts)
{
  std::atomic<bool> exit(false);
  cv::Mat image;
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  string const face_xml = "./face.xml";
  //string const face_xml = "../opencv/data/haarcascades/haarcascade_frontalface_alt2.xml";

  vector<AlphaImage> hats;
  hats.emplace_back("sombrero.png");

  std::vector<std::thread> workers;

  ThreadSafeMat opt_flow(cv::Mat::zeros(stream.height(), stream.width(), CV_8UC3));

  if (opts.optical_flow) {
    workers.emplace_back(optical_flow_thread,
                         std::ref(stream), std::ref(opt_flow), std::ref(exit));
  }

  if (opts.face_detect) {
    workers.emplace_back(facerecognition_thread,
                         std::ref(stream), face_xml,
                         std::ref(hats[0]), std::ref(exit));
  }

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

    if (opts.optical_flow) {
      cv::imshow("OptFlow", opt_flow.get());
    }

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

  for (auto &t : workers) {
    t.join();
  }
}

// returns processed arguments
int check_options(Options &opts, int const argc, char const * const *argv)
{
  int i = 0;
  argv++;
  for (; i < argc; i++) {
    std::string arg(argv[i]);
    std::cout << "processing " << i << "/" << argc << ": " << arg << std::endl;

    if (arg.find_first_of("-") == std::string::npos) {
      break;
    }

    if (arg == "-w" || arg == "--width") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
      }
      opts.width = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-h" || arg == "--height") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
      }
      opts.height = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-c" || arg == "--camera") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
      }
      opts.cam_num = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-f" || arg == "--face-detect") {
      opts.face_detect = true;
    } else if (arg == "-o" || arg == "--optical-flow") {
      opts.optical_flow = true;
    } else if (arg == "-h" || arg == "--help") {
      return -1;
    } else {
      std::cerr << "unknown option: " << arg << std::endl;
      return -1;
    }
  }

  return i;
}

int main(int argc, char **argv)
{
  Options opts;
  int nopts = check_options(opts, argc, argv);
  if (nopts == -1) {
    usage(argv[0]);
    return 1;
  }
  argc -= nopts;

  std::cout << "Options: " << std::endl << opts;

  //LiveStream live(cam);
  //LiveStream live(cam, 1920, 1080);
  //LiveStream live(cam, 1280, 720);
  LiveStream live(opts.cam_num, opts.width, opts.height);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << opts.cam_num << endl;
    return -1;
  }

  capture_loop(live, opts);

  return 0;
}
