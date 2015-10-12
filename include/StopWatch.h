#ifndef STOPWATCH_H_INCLUDED
#define STOPWATCH_H_INCLUDED

#include <opencv2/core.hpp>
#include <iostream>

class StopWatch {

protected:
  double mStart;
  double mStop;

  bool mRunning = false;

  friend std::ostream &operator<<(std::ostream &out, StopWatch const &sw);

public:
  StopWatch()
  {
    start();
  }

  inline void start()
  {
    mRunning = true;
    mStart = cv::getTickCount();
  }

  inline void stop()
  {
    mStop = cv::getTickCount();
    mRunning = false;
  }

  inline double getDurationMs() const
  {
    return (mStop - mStart) / cv::getTickFrequency() * 1000;
  }
};

inline std::ostream &operator<<(std::ostream &out, StopWatch const &sw)
{
  out << sw.getDurationMs() << "ms";
  return out;
}

class ScopedStopWatch : public StopWatch {
private:
  std::string mText;

public:
  ScopedStopWatch(std::string const &text) : StopWatch(), mText(text) { }
  virtual ~ScopedStopWatch()
  {
    stop();
    std::cout << mText << " " << *this << std::endl;
  }
};

#endif
