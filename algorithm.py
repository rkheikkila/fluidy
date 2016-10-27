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


def register_images(fixed_image, moving_image, initial_params=None, sample_ratio=1.0, lbfgs=True, multires=True):
    """Performs non-rigid image registration on a collection of images.

    Args:
        fixed_image: the image that is considered fixed in the registration process
        moving_image: image to be transformed
        initial_params: preoptimized parameters used as an initial guess
        sample_ratio: ratio of pixels sampled in calculation of sum of squares difference.
        lbfgs: flag to enable the LBFGS optimization algorithm. If false,
                gradient descent method is used instead.
        multires: flag to enable multi-resolution framework, which results in a more robust optimization process.

    Returns:
        the optimized non-rigid transformation
    """
    max_iter = 100
    grid_physical_spacing = [10.0, 10.0]
    registration = sitk.ImageRegistrationMethod()

    # Initialize B-spline transformation and control point grid
    image_physical_size = [size * spacing for size, spacing in zip(fixed_image.GetSize(), fixed_image.GetSpacing())]
    mesh_size = [int(img_size / grid_spacing + 0.5)
                    for img_size, grid_spacing in zip(image_physical_size, grid_physical_spacing)]
    initial_transform = sitk.BSplineTransformInitializer(image1=fixed_image,
                                                         transformDomainMeshSize=mesh_size, order=3)
    if initial_params:
        initial_transform.SetParameters(initial_params)

    registration.SetInitialTransform(initial_transform)

    registration.SetMetricAsMeanSquares()
    if 0 < sample_ratio < 1.0:
        registration.SetMetricSamplingStrategy(registration.RANDOM)
        registration.SetMetricSamplingPercentage(sample_ratio)

    if multires:
        registration.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        registration.SetSmoothingSigmasPerLevel(smoothingSigmas=[8, 4, 0])
        registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    if lbfgs:
        registration.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=max_iter)
    else:
        registration.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=max_iter)

    registration.SetInterpolator(sitk.sitkLinear)
    return registration.Execute(fixed_image, moving_image)


def iterative_registration(image_arrays, max_iter=5):
    """Performs iterative registration for a set of frames.

    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    # TODO: Develop a workaround for RGB images
    greyscale_images = (image.mean(axis=2) for image in image_arrays)
    frames = [sitk.GetImageFromArray(frame) for frame in greyscale_images]
    # Calculate mean by exploiting the overloaded addition operator
    # Can this summation overflow? (pixel values are 64-bit float)
    temporal_mean = sum(frames) / len(frames)

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(temporal_mean.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    means = [temporal_mean]

    # TODO: Implement a convergence criterion based on sum-of-squares metric
    for i in range(max_iter):
        resampler.SetReferenceImage(temporal_mean)

        transformed_frames = []
        for frame in frames:
            # TODO: Blur frames to improve registration quality
            # Apply non-rigid registration
            transformation = register_images(temporal_mean, frame)
            # Dewarp images using the optimized transformation
            resampler.SetTransform(transformation)
            transformed_frame = resampler.Execute(frame)
            transformed_frames.append(transformed_frame)

        temporal_mean = sum(transformed_frames) / len(transformed_frames)
        means.append(temporal_mean)
        frames = transformed_frames

    return [sitk.GetArrayFromImage(mean) for mean in means]


#   Testing	
if __name__ == "__main__":
    frames = load_frames('expdata_middle.avi')
    imgs = iterative_registration(frames)
    for i in range(len(imgs)): save_image(imgs[i], "result{}.jpg".format(i))
