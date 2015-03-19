#include <condition_variable>
#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <atomic>
#include <unistd.h>
#include <sys/syscall.h>

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/cuda.hpp"
#include "opencv2/cudaoptflow.hpp"

#include "augmented-reality.h"
#include "facedetection.h"
#include "optical-flow.h"
#include "util.h"

using namespace std;
using namespace cv;

using timepoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct Options {
  int cam_num = 0;
  int width = 640;
  int height = 480;
  bool face_detect = false;
  bool augmented_reality = false;
  bool optical_flow = false;
  std::string face_xml = "face.xml";
};

std::ostream &operator<<(ostream &out, Options const &o)
{
  out << "Camera:            " << o.cam_num << std::endl
      << "Width:             " << o.width << std::endl
      << "Height:            " << o.height << std::endl
      << "Facedetect:        " << std::boolalpha << o.face_detect << std::endl
      << "Augmented Reality: " << std::boolalpha << o.augmented_reality << std::endl
      << "Optical Flow:      " << std::boolalpha << o.optical_flow << std::endl
      << "Haarcascade XML:   " << o.face_xml << std::endl;
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
            << " -f, --face-detect: Enable face detection" << std::endl
            << "                    (needed for augmented reality and face visualization of optical flow" << std::endl
            << " -a, --augmented-reality: Enable augmented reality" << std::endl
            << " -o, --optical-flow: Enable optical flow analysis" << std::endl
            << " -x, --face-xml: XML file containing haarcascade for face detection" << std::endl
            << " --help: Show this help" << std::endl
            << std::endl;
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
    } else if (arg == "-x" || arg == "--face-xml") {
      if ((i + 1) >= argc) {
        std::cerr << "missing value for " << arg << std::endl;
        return -1;
      }
      opts.face_xml = std::string(argv[i + 1]);
      i++;
    } else if (arg == "-f" || arg == "--face-detect") {
      opts.face_detect = true;
    } else if (arg == "-a" || arg == "--augmented-reality") {
      opts.augmented_reality = true;
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

class ConditionalWait {
private:
  std::atomic<bool> &mExit;

  std::mutex mMutex;
  std::condition_variable mEvent;
  bool mFlag;

public:
  ConditionalWait(std::atomic<bool> &exit, bool init) : mExit(exit), mFlag(init) { }
  operator bool() { return mFlag; }

  void set() {
    mFlag = true;
    notify();
  }

  void clear() {
    mFlag = false;
    notify();
  }

  void toggle() {
    mFlag = !mFlag;
    notify();
  }

  void wait() {
    if (mFlag) return;

    std::unique_lock<std::mutex> l(mMutex);
    mEvent.wait(l, [&](){return mExit || mFlag;});
  }

  void notify() {
    mEvent.notify_all();
  }
};

void capture_loop(LiveStream &stream, Options opts)
{
  std::atomic<bool> exit(false);
  cv::Mat image;

  vector<AlphaImage> hats;
  Faces faces;
  //FaceDetection<cv::cuda::CascadeClassifier_CUDA> facedetection(stream, faces, opts.face_xml);
  FaceDetection<cv::CascadeClassifier> facedetection(stream, faces, opts.face_xml);
  if (!facedetection.isReady()) {
    std::cerr << "loading FaceDetection failed" << std::endl;
    return;
  }
  std::cout << "FaceDetection loaded" << std::endl;

  AugmentedReality ar(stream, &faces);
  ar.addHat("sombrero.png", 2, 4);
  ar.addHat("tophat.png", 1.2, 10);
  ar.addHat("crown.png", 1.2, 16);
  ar.addHat("fancy.png", 2, 4);
  if (!ar.ready()) {
    std::cerr << "loading AugmentedReality failed" << std::endl;
    return;
  }
  std::cout << "AugmentedReality loaded" << std::endl;

  ThreadSafeMat of_visualize(cv::Mat::zeros(stream.height(), stream.width(), CV_8UC3));
  OpticalFlow of(stream, of_visualize);
  of.setFaces(&faces);
  if (!of.isReady()) {
    std::cerr << "loading OpticalFlow failed" << std::endl;
    return;
  }
  std::cout << "OpticalFlow loaded" << std::endl;

  ConditionalWait face_wait(exit, opts.face_detect);
  ConditionalWait ar_wait(exit, opts.augmented_reality);
  ConditionalWait of_wait(exit, opts.optical_flow);
  std::vector<std::thread> workers;

  std::cout << "PID main thread: " << syscall(SYS_gettid) << std::endl;

  double face_time, ar_time, of_time;

  workers.emplace_back([&facedetection, &ar, &exit, &ar_wait, &face_wait, &face_time, &ar_time]()
                       {
                        std::cout << "PID face detection / augmented reality thread: " << syscall(SYS_gettid) << std::endl;
                        while(!exit) {
                          face_wait.wait();

                          double t = (double) cv::getTickCount();
                          facedetection.detect();
                          face_time = ((double) cv::getTickCount() - t) / getTickFrequency();

                          if (ar_wait) {
                            double t = (double) cv::getTickCount();
                            ar();
                            ar_time = ((double) cv::getTickCount() - t) / getTickFrequency();
                          }
                        }
                       });

  workers.emplace_back([&of, &exit, &of_wait, &of_time]()
                       {
                        std::cout << "PID optical flow thread: " << syscall(SYS_gettid) << std::endl;
                        while(!exit) {
                          of_wait.wait();
                          double t = (double) cv::getTickCount();
                          of();
                          of_time = ((double) cv::getTickCount() - t) / getTickFrequency();
                        }
                       });

  const std::string live_feed_window = "Live Feed";
  const std::string opt_flow_window = "Optical Flow";
  bool opt_flow_result = true;
  bool live_feed = true;
  /*
  cv::namedWindow(live_feed_window, CV_WINDOW_NORMAL);
  cv::setWindowProperty(live_feed_window, CV_WND_PROP_FULLSCREEN, CV_WND_PROP_FULLSCREEN);
  cv::moveWindow(live_feed_window, 0, 0);
  cv::resizeWindow(live_feed_window, 1920, 1080);
  */

  double time = 0;

  while (!exit) {

    double t = (double) cv::getTickCount();
    // take new image
    stream.nextFrame(image);

    if (opt_flow_result) {
      cv::imshow(opt_flow_window, of_visualize.get());
    }

    if (live_feed) {
      stream.applyOverlay(image);
      double total = ((double) getTickCount() - t) / getTickFrequency();

      std::vector<PrintableTime> times =
      {
        { "facedetect: ", &face_time },
        { "ar:         ", &ar_time },
        { "opt flow:   ", &of_time },
        { "total:      ", &total },
      };

      cv::Point pos = print_times(image, cv::Point(50, 50), times);

      double frame_rate = ((double) getTickCount() - time) / getTickFrequency();
      std::stringstream ss;
      ss << "FPS: " << 1/frame_rate;
      cv::putText(image, ss.str(), pos, FONT_HERSHEY_PLAIN, 1.2, Scalar(128, 255, 255));

      cv::imshow(live_feed_window, image);
    }

    time = (double) getTickCount();

    // check for button press for 30ms. necessary for opencv to refresh windows
    char key = cv::waitKey(5);
    switch (key) {
      case 'q':
        exit = true;
        ar_wait.notify();
        of_wait.notify();
        face_wait.notify();
        break;
      case 'l':
        live_feed = !live_feed;
        cv::destroyWindow(live_feed_window);
        break;
      case 'k':
        opt_flow_result = !opt_flow_result;
        cv::destroyWindow(opt_flow_window);
        break;
      case 'v':
        of.toggle_visualization();
        break;
      case 'o':
        of_wait.toggle();
        std::cout << "OpticalFlow: " << (of_wait ? "enabled" : "disabled") << std::endl;
        of_visualize.update(cv::Mat::zeros(stream.height(), stream.width(), CV_8UC3));
        of_time = 0;
        break;
      case 'f':
        face_wait.toggle();
        std::cout << "FaceDetection: " << (face_wait ? "enabled" : "disabled") << std::endl;
        face_time = 0;
        break;
      case 'a':
        ar_wait.toggle();
        std::cout << "AugmentedReality: " << (ar_wait ? "enabled" : "disabled") << std::endl;
        stream.resetOverlay();
        ar_time = 0;
        break;
      default:
        break;
    }
  }

  for (auto &t : workers) {
    t.join();
  }
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

  int gpu = 0;
  cv::cuda::setDevice(gpu);
  cv::cuda::resetDevice();
  cv::cuda::DeviceInfo info;
  std::cout << "using GPU" << gpu << ": "
            << info.freeMemory() / 1024 / 1024 << " / "
            << info.totalMemory() / 1024 / 1024 << " MB in use" << std::endl;

  LiveStream live(opts.cam_num, opts.width, opts.height);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << opts.cam_num << endl;
    return -1;
  }

  capture_loop(live, opts);

  return 0;
}
