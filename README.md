# tegra_tdot
Demo application for NVIDIA Tegra

<p>
This demo application was developed for a showcase at the open day of the Upper Austria
Univeristy of Applied Sciences faculty of Hagenberg, department of Embedded System Design.
It should illustrate the use of the OpenCV library on the NVIDIA Tegra K1 development kit.

It implements an edge detection, optical flow analysis and a face detection with augmented realitiy using the
OpenCV 3 API.
</p>
## Compiling
The provided Makefile should suffice to build the application. The follwing libraries are needed
- OpenCV 3.0.0-beta (I built the library from source from tag: 3.0.0-beta (hash: ae4cb57), please refer to READMEs and tutorials for instructions on how to build it for your platform)
- CUDA Toolbox 6.5

The makefile is hardcoded to look for the libraries in /opt but should be easily modified to suit your needs.

## Running
Before running the application, make sure you have the necessary libraries in your libary search path, or add them temporarily using
```
export LD_LIBRARY_PATH=/opt/opencv3/lib:/opt/cuda-6.5/lib64
```
and setting the environment variable accroding to your system.

Consider running
```
./tdot_demo --help
```
to see all available options.
While running the application keyboard shortcuts can be used to (de-)activate certain visualizations:
- 'e' toggles the edge detection window
- 'o' toggles the optical flow calculation
- 'v' cycles through the optical flow visualizations
  - 'Arrows': shows green, red and white arrows according to the motion, where the length of the error implies the speed of the object
  - 'Blocks': green and red blocks<
  - 'Faces': shows green and red squares where faces are detected and indicates there relative motion.</br>
    Note that face detection has to be enabled to see the squares
- 'f' toggles face detection
- 'a' toggles augmented reality that draws hats on each detected face</br>
  Note that face detection has to be enabled to see the hats
- 'l' toggles the live view window
- 'k' toggles the optical flow window
- 'q' closes all windows and quits the application
