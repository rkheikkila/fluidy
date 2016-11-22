# fluidy
Image processing for seeing through water.

A simple example is given below (sample video frame on the left, our result on the right):

![Sample frame](examples/brick_sample.png?raw=true) ![Result](examples/brick.jpg?raw=true)

Example data from [Carnegie Mellon University](http://www.cs.cmu.edu/~ILIM/projects/IM/water/research_water.html).

This algorithm reduces distortions in a video caused by a wavy water surface. Our algorithms are based on two different iterative non-rigid registration methods:

* [Oreifej et al. 2011: A Two-Stage Reconstruction Approach for Seeing Through Water](http://www.cs.ucf.edu/~oreifej/papers/WATER_CVPR2011.pdf)
* [Halder et al. 2014: High accuracy image restoration method for seeing through water](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1910348)

However, instead of using non-rigid registration we dewarp images by calculating pixel shift maps using optical flow.

Developed by Fluidy, Aalto University 2016.

# Dependencies
- [numpy](http://www.numpy.org/)
- [OpenCV](http://www.opencv.org/)
- Robust PCA implementation by [Kyle Kastner](http://kastnerkyle.github.io/posts/robust-matrix-decomposition/)

**NOTE**: Only Python 2.7 and OpenCV 2.4.13 are supported due to API changes in OpenCV 3.

