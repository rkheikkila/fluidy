"""
Image processing algorithm for seeing through water.

This algorithm reduces distortions in a video caused by water surface and
produce a clear image of the plane below the water surface.
"""

import numpy as np
import SimpleITK as sitk
import cv2


def load_frames(filename):
    """Loads a video file and converts the frames to a numpy array.

    Returns:
        a list of 3D numpy arrays containing the video frames.
    """
    vidcap = cv2.VideoCapture(filename)
    return [vidcap.read()[1] for i in range(int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))]	


def save_image(array, filename):
    """ Saves a image array to disk

    Args:
        array: a 3D numpy array containing the image
        filename: name of image file

    Returns:
        nothing, saves image to disk
    """
    cv2.imwrite(filename,array)


def register_images(frames):
    """Performs non-rigid image registration on a collection of images.

    Args:
        frames: a list of 3D numpy arrays containing the images.

    Returns:
        the applied non-rigid transformation
    """
    sitk_frames = [sitk.GetImageFromArray(frame, isVector=True) for frame in frames]
    image = sitk.JoinSeries(sitk_frames)

    registration = sitk.ImageRegistrationMethod()

    # Initialize B-Spline transformation
    grid_physical_spacing = [50.0, 50.0, 50.0]
    image_physical_size = [size*spacing for size,spacing in zip(image.GetSize(), image.GetSpacing())]
    mesh_size = [int(img_size/grid_spacing + 0.5)
                 for img_size,grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    initial_transform = sitk.BSplineTransformInitializer(image1 = image,
                                                         transformDomainMeshSize = mesh_size, order=3)
    registration.SetInitialTransform(initial_transform)

    # Use sum of squares as optimization metric
    registration.SetMetricAsMeanSquares()
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.01)

    # Multi-resolution framework.
    registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration.SetInterpolator(sitk.sitkLinear)

    # Optimize using gradient descent method
    registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)

    return registration.Execute(image, image)


#   Testing	
if __name__ == "__main__":
	frames = load_frames('Shortest Video on Youtube.mp4')
	register_images(frames)
	[save_image(frames[i],'images/' + str(i) + '.jpeg') for i in range(len(frames))]