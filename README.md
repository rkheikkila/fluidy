# fluidy
Image processing for seeing through water.

This algorithm reduces distortions in a video caused by a wavy water surface. Our algorithms are based on two different iterative non-rigid registration methods:

* [Oreifej et al. 2011: A Two-Stage Reconstruction Approach for Seeing Through Water](http://www.cs.ucf.edu/~oreifej/papers/WATER_CVPR2011.pdf)
* [Halder et al. 2014: High accuracy image restoration method for seeing through water](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=1910348)

However, instead of utilizing non-rigid registration we calculate shift maps using optical flow.

Developed by Fluidy, Aalto University 2016.

# Dependencies
- [numpy](http://www.numpy.org/)
- [OpenCV](http://www.opencv.org/)
- Robust PCA implementation by [Kyle Kastner](http://kastnerkyle.github.io/posts/robust-matrix-decomposition/)


