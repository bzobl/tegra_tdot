#ifndef FACEDETECTION_H_INCLUDED
#define FACEDETECTION_H_INCLUDED

#include <mutex>
#include <vector>
#include <algorithm>
#include <cassert>
#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/cuda.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include "faces.h"
#include "livestream.h"
#include "util.h"

template <typename TCascade = cv::cuda::CascadeClassifier_CUDA>
class FaceDetection {

private:

  LiveStream &mStream;
  Faces &mFaces;
  TCascade mFaceCascade;

protected:
  const double SCALE_FACTOR = 1.2;
  const int MIN_NEIGHBOURS = 4;
  const cv::Size MIN_SIZE = cv::Size(60, 60);

  void do_facedetection(cv::Mat const &frame);

public:
  FaceDetection(LiveStream &stream, Faces &faces, std::string const &face_cascade);

  bool isReady();
  void detect();
};

template <typename TCascade>
FaceDetection<TCascade>::FaceDetection(LiveStream &stream,
                                               Faces &faces,
                                               std::string const &face_cascade)
                                               : mStream(stream),
                                                 mFaces(faces),
                                                 mFaceCascade(face_cascade)
{
}

template <typename TCascade>
void FaceDetection<TCascade>::do_facedetection(cv::Mat const &frame)
{
  std::cerr << "###" << std::endl;
  std::cerr << "Face detection not implemented!!!" << std::endl;
  std::cerr << "###" << std::endl;
}

template <>
void FaceDetection<cv::cuda::CascadeClassifier_CUDA>::do_facedetection(cv::Mat const &frame)
{
  cv::Mat h_faces;
  cv::cuda::GpuMat d_frame, d_faces;
  d_frame.upload(frame);

  int n_detected = mFaceCascade.detectMultiScale(d_frame, d_faces,
                                                 SCALE_FACTOR, MIN_NEIGHBOURS, MIN_SIZE);
  
  d_faces.colRange(0, n_detected).download(h_faces);
  cv::Rect *prect = h_faces.ptr<cv::Rect>();

  std::unique_lock<std::mutex> l(mFaces.getMutex());
  for (int i = 0; i < n_detected; i++) {
    mFaces.addFace(prect[i]);
  }
}

template <>
void FaceDetection<cv::CascadeClassifier>::do_facedetection(cv::Mat const &frame)
{
  std::vector<cv::Rect> faces;
  mFaceCascade.detectMultiScale(frame, faces, SCALE_FACTOR, MIN_NEIGHBOURS, 0, MIN_SIZE);

  std::unique_lock<std::mutex> l(mFaces.getMutex());
  for (cv::Rect &face : faces) {
    mFaces.addFace(face);
  }
}

template <typename TCascade>
bool FaceDetection<TCascade>::isReady()
{
  return mStream.isOpened() && !mFaceCascade.empty();
}

template <typename TCascade>
void FaceDetection<TCascade>::detect()
{
  assert(isReady());

  cv::Mat frame;
  std::vector<cv::Rect> faces;

  double start, mutex_locked, tick_done, got_frame, detection_done;

  start = (double) cv::getTickCount();

  mutex_locked = (double) cv::getTickCount();

  // update ttl of all faces
  mFaces.tick();

  tick_done = (double) cv::getTickCount();

  mStream.getFrame(frame);
  cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

  got_frame = (double) cv::getTickCount();

  do_facedetection(frame);

  detection_done = (double) cv::getTickCount();

  std::string debug_window_name = "Debug Window";
  cv::Mat debug = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);
  double mutex = (mutex_locked - start) / cv::getTickFrequency();
  double tick = (tick_done - mutex_locked) / cv::getTickFrequency();
  double frame_t = (got_frame - tick_done) / cv::getTickFrequency();
  double detection = (detection_done - got_frame) / cv::getTickFrequency();
  double total = (detection_done - start) / cv::getTickFrequency();

  std::vector<PrintableTime> times =
  {
    { "mutex:     ", &mutex },
    { "tick;      ", &tick },
    { "get frame: ", &frame_t },
    { "detection: ", &detection },
    { "total:     ", &total },
  };

  print_times(debug, cv::Point(50, 50), times);
  //cv::imshow(debug_window_name, debug);
}

#endif
