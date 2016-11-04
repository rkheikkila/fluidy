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


def mean_of_differences(image1, image2):
    """Calculates the sum of absolute pixelwise differences of two images.

    Args:
        image1, image2: two image arrays of same shape to compare
    Returns:
        a tuple containing mean differences for each channel
    """
    # Rescale the image intensities to [0,1]
    scaledImage1 = cv2.normalize(image1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    scaledImage2 = cv2.normalize(image2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    differences = cv2.absdiff(scaledImage1, scaledImage2)
    return cv2.mean(differences)


def warp_flow(img, flow):
    """Applies a generic transformation to an image
    
    Args: 
        img: image to transform
        flow: transformation array  
    Returns:
        the transformed image
    """    
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow.astype('float32'), None, cv2.INTER_LINEAR)
    return res


def optical_flow(prev, next):
    """Calculates the dense optical flow from one image to another
    
    Args: 
        prev: initial image
        next: shifted image 
    Returns:
        the movement of each pixel between the images
    """
    # TODO: Use previous flow array as an initial guess
    flow = cv2.calcOpticalFlowFarneback(prev, next, pyr_scale = 0.5, levels = 3, winsize = 15, iterations = 3, poly_n = 5, poly_sigma = 1.2, flags = cv2.OPTFLOW_USE_INITIAL_FLOW, ) #  , cv2.cv.OPTFLOW_USE_INITIAL_FLOW
    return flow


def hta_algorithm(image_arrays, max_iter=5):
    """Performs iterative registration for a set of frames.

    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    # This method doesn't work yet
    frames = [image.mean(axis=2) for image in image_arrays]
    temporal_mean = sum(frames) / len(frames)
    h, w = temporal_mean.shape[:2]

    means = []
    means.append(temporal_mean)

    for i in range(max_iter):
        shift_maps = []
        shift_maps.append(np.zeros((h,w,2)))
        for i in range(len(frames)-1):
            shift_maps.append(optical_flow(frames[0],frames[i+1]))

        centroid_shift_map = sum(shift_maps) / len(shift_maps)

        corrected_shift_maps = []
        for n in range(len(shift_maps)):
            map = centroid_shift_map
            for i in range(w):
                for j in range(h):
                    indx = max(min(i + int(round(w*map[j,i,0])),w-1),0)
                    indy = max(min(i + int(round(h*map[j,i,1])),h-1),0)
                    map[j,i,:] = map[j,i,:] + shift_maps[n][indy,indx,:]
            corrected_shift_maps.append(map)

        dewarped_frames = [warp_flow(frames[i],shift_maps[i]) for i in range(len(frames))]
        temporal_mean = sum(dewarped_frames) / len(dewarped_frames)
        means.append(temporal_mean)
        frames = dewarped_frames

    return means


def oreifej_algorithm(image_arrays, max_iter=10):
    """Performs iterative registration for a set of frames.

    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    # TODO: Blur the frames for a better result
    convergence_threshold = 0.01
    frames = [image.mean(axis=2) for image in image_arrays]
    temporal_mean = sum(frames) / len(frames)

    means = []
    means.append(temporal_mean)

    for iter in range(max_iter):
        shift_maps = [optical_flow(frames[i], temporal_mean) for i in range(len(frames))]
        dewarped_frames = [warp_flow(frames[i], shift_maps[i]) for i in range(len(frames))]

        new_mean = sum(dewarped_frames) / len(dewarped_frames)
        diffs = mean_of_differences(temporal_mean, new_mean)
        temporal_mean = new_mean
        means.append(temporal_mean)
        frames = dewarped_frames

        if diffs[0] < convergence_threshold:
            break

    return means


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
    temporal_mean = sum(frames) / len(frames)

    resampler = sitk.ResampleImageFilter()
    resampler.SetDefaultPixelValue(temporal_mean.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear)

    means = [temporal_mean]

    # TODO: Implement a convergence criterion based on sum-of-squares metric
    for i in range(max_iter):
        resampler.SetReferenceImage(temporal_mean)
        transformed_frames = []

        # Compute transformation for the first frame and use it as an initial guess with next
        first_transformation = register_images(temporal_mean, frames[0])
        # Apply non-rigid registration
        params = first_transformation.GetParameters()
        resampler.SetTransform(first_transformation)
        # Dewarp images using the optimized transformation
        transformed_frame = resampler.Execute(frames[0])
        transformed_frames.append(transformed_frame)

        # Compute the rest
        for frame in frames[1:]:
            # TODO: Blur frames to improve registration quality
            transformation = register_images(temporal_mean, frame,
                                             initial_params=params, multires=False)
            params = transformation.GetParameters()
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
    #imgs = iterative_registration(frames[0:99])
    imgs = oreifej_algorithm(frames[0:99])
    for i in range(len(imgs)): save_image(imgs[i], "result{}.jpg".format(i))
