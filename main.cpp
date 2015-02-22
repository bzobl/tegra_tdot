#include <condition_variable>
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
#include "faces.h"
#include "optical-flow.h"

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
};

std::ostream &operator<<(ostream &out, Options const &o)
{
  out << "Camera:            " << o.cam_num << std::endl
      << "Width:             " << o.width << std::endl
      << "Height:            " << o.height << std::endl
      << "Facedetect:        " << std::boolalpha << o.face_detect << std::endl
      << "Augmented Reality: " << std::boolalpha << o.augmented_reality << std::endl
      << "Optical Flow:      " << std::boolalpha << o.optical_flow << std::endl;
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
  std::mutex mMutex;
  std::condition_variable mEvent;
  bool mFlag;

  std::atomic<bool> &mExit;

public:
  ConditionalWait(std::atomic<bool> &exit, bool init) : mExit(exit), mFlag(init) { }
  void toggle() { mFlag = !mFlag; };
  operator bool() { return mFlag; }

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
  cv::Mat lastGrayscale, nowGrayscale;
  cv::Mat diff, thresh;

  string const face_xml = "./face.xml";

  vector<AlphaImage> hats;
  Faces faces(stream, face_xml);
  if (!faces.isReady()) {
    std::cerr << "loading FaceDetection failed" << std::endl;
    return;
  }
  std::cout << "FaceDetection loaded" << std::endl;

  AugmentedReality ar(stream, &faces);
  ar.addHat("sombrero.png");
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
  std::vector<std::thread> workers;

  workers.emplace_back([&faces, &exit, &face_wait]()
                       {
                        while(!exit) {
                          face_wait.wait();
                          faces.detect();
                          std::cout << "detecting faces" << std::endl;
                        }
                       });

  if (opts.face_detect) {
    workers.emplace_back([&ar, &exit, &opts]()
                         {
                          while(!exit) {
                            if (opts.augmented_reality) {
                              ar();
                            }
                          }
                         });
  }

  if (opts.optical_flow) {
    workers.emplace_back([&of, &exit, &opts]()
                         {
                          while(!exit) {
                            if (opts.optical_flow) {
                              of();
                            }
                          }
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
      case 'v':
        of.toggle_visualization();
        break;
      case 'o':
        opts.optical_flow = !opts.optical_flow;
        std::cout << "OpticalFlow: " << (opts.optical_flow ? "enabled" : "disabled") << std::endl;
        break;
      case 'f':
        face_wait.toggle();
        std::cout << "FaceDetection: " << (face_wait ? "enabled" : "disabled") << std::endl;
        break;
      case 'a':
        opts.augmented_reality = !opts.augmented_reality;
        std::cout << "AugmentedReality: " << (opts.augmented_reality ? "enabled" : "disabled") << std::endl;
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

  LiveStream live(opts.cam_num, opts.width, opts.height);
  if (!live.isOpened()) {
    cerr << "Error opening camera " << opts.cam_num << endl;
    return -1;
  }

  capture_loop(live, opts);

  return 0;
}
