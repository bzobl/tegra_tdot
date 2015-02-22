#include "augmented-reality.h"

AugmentedReality::AugmentedReality(LiveStream &stream, std::string const &face_cascade)
                                  : mStream(stream), mFaceCascade(face_cascade)
{

}

void AugmentedReality::addHat(std::string const &file)
{
  mHats.emplace_back(file);
}

bool AugmentedReality::ready()
{
  return mStream.isOpened() && !mFaceCascade.empty() && !mHats.empty();
}

std::vector<cv::Rect> AugmentedReality::detect_faces(cv::Mat &frame)
{
  std::vector<cv::Rect> faces;
  cv::cuda::GpuMat d_faces;
  cv::Mat h_faces;

  cv::Mat gray;
  cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  cv::cuda::GpuMat d_gray(gray);

  int n_detected = mFaceCascade.detectMultiScale(d_gray, d_faces, 1.2, 8, cv::Size(40, 40));

  d_faces.colRange(0, n_detected).download(h_faces);
  cv::Rect *prect = h_faces.ptr<cv::Rect>();

  for (int i = 0; i < n_detected; i++) {
    faces.push_back(prect[i]);
  }

  return faces;
}

void AugmentedReality::operator()()
{
  assert(ready());

  cv::Mat frame;
  mStream.getFrame(frame);
  std::vector<cv::Rect> faces = detect_faces(frame);

  // for the duration of resetting the overlay no other thread must use the overlay
  {
    std::unique_lock<std::mutex> l(mStream.getOverlayMutex());
    mStream.resetOverlay();
    for (cv::Rect face : faces) {
      AlphaImage &hat(mHats[0]); 

      int width = face.width * 2;
      int height = hat.height(width);
      int x = face.x - width/4;
      int y = face.y - height;
      mStream.addImageToOverlay(hat, width, x, y);
    }
  }
}
