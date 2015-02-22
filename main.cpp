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

#include "augmented-reality.h"
#include "optical-flow.h"

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

std::ostream &operator<<(ostream &out, Options const &o)
{
  out << "Camera:       " << o.cam_num << std::endl
      << "Width:        " << o.width << std::endl
      << "Height:       " << o.height << std::endl
      << "Facedetect:   " << std::boolalpha << o.face_detect << std::endl
      << "Optical Flow: " << std::boolalpha << o.optical_flow << std::endl;
  return out;
}

void usage(char const * const progname)
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
            << " --help: Show this help" << std::endl
            << std::endl;
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
  AugmentedReality ar(stream, face_xml);
  ar.addHat("sombrero.png");
  std::cout << "AugmentedReality loaded" << std::endl;

  ThreadSafeMat of_visualize(cv::Mat::zeros(stream.height(), stream.width(), CV_8UC3));
  OpticalFlow of(stream, of_visualize);

  std::vector<std::thread> workers;

  if (opts.face_detect) {
    workers.emplace_back([&ar, &exit]()
                         {
                          while(!exit) { ar(); }
                         });
  }

  if (opts.optical_flow) {
    workers.emplace_back([&of, &exit]()
                         {
                          while(!exit) { of(); }
                         });
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
      cv::imshow("OptFlow", of_visualize.get());
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
  int i = 1;
  for (; i < argc; i++) {
    std::string arg(argv[i]);

    if (arg.find_first_of("-") == std::string::npos) {
      break;
    }

    if (arg == "-w" || arg == "--width") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
        return -1;
      }
      opts.width = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-h" || arg == "--height") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
        return -1;
      }
      opts.height = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-c" || arg == "--camera") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
        return -1;
      }
      opts.cam_num = atoi(argv[i + 1]);
      i++;
    } else if (arg == "-f" || arg == "--face-detect") {
      opts.face_detect = true;
    } else if (arg == "-o" || arg == "--optical-flow") {
      opts.optical_flow = true;
    } else if (arg == "--help") {
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

  LiveStream live(opts.cam_num, opts.width, opts.height);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << opts.cam_num << endl;
    return -1;
  }

  capture_loop(live, opts);

  return 0;
}
