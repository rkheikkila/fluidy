# fluidy
Image processing for seeing through water.

Two simple examples are given below (sample video frame on the left, followed by results from two different algorithms):

![Sample frame](examples/brick_sample.png?raw=true) ![Oreifej result](examples/brick.jpg?raw=true) ![HTA result](examples/brick_hta.jpg?raw=true)

![Sample frame](examples/checkboard_sample.png?raw=true) ![Oreifej result](examples/checkboard.jpg?raw=true) ![HTA result](examples/checkboard_hta.jpg?raw=true)

Example data from [Carnegie Mellon University](http://www.cs.cmu.edu/~ILIM/projects/IM/water/research_water.html).

These algorithms reduce distortions in a video caused by a wavy water surface. The algorithms are based on two different iterative non-rigid registration methods:

* [Oreifej et al. 2011: A Two-Stage Reconstruction Approach for Seeing Through Water](http://www.cs.ucf.edu/~oreifej/papers/WATER_CVPR2011.pdf)
* [Halder et al. 2014: High accuracy image restoration method for seeing through water](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1910348)

However, instead of using non-rigid registration we dewarp images by calculating pixel shift maps using optical flow.

Developed by Team Fluidy, Aalto University 2016.

# Dependencies
- [numpy](http://www.numpy.org/)
- [OpenCV](http://www.opencv.org/)
- Robust PCA implementation by [Kyle Kastner](http://kastnerkyle.github.io/posts/robust-matrix-decomposition/)

**NOTE**: Only Python 2.7 and OpenCV 2.4.13 are supported due to API changes in OpenCV 3.

