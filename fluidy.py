"""
Image processing algorithm for seeing through water.

This algorithm reduces distortions in a video caused by water surface and
produce a clear image of the plane below the water surface.
"""

from numpy.linalg import norm, svd

import cv2
import numpy as np

import os


def load_frames(filename):
    """Loads a video file and converts the frames to a numpy array.

    Returns:
        a list of 3D numpy arrays containing the video frames.
    """
    vidcap = cv2.VideoCapture(filename)
    vidcap.set(cv2.cv.CV_CAP_PROP_FORMAT, cv2.CV_8U)
    return [vidcap.read()[1] for i in range(int(vidcap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)))]	


def save_video(frames, filename):
    """ Saves a video to disk

    Args:
        frames: a list of 3D numpy arrays
        filename: name of video file

    Returns:
        nothing, saves video to disk
    """
    h, w = frames[0].shape[:2]
    vidwrite = cv2.VideoWriter(filename, cv2.cv.CV_FOURCC('D','I','V','X'), fps = 30.0, frameSize = (w, h))
    for frame in frames:
        vidwrite.write(frame)


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
    flow = cv2.calcOpticalFlowFarneback(prev, next, pyr_scale=0.5, levels=3,
                                        winsize=15, iterations=3, poly_n=5,
                                        poly_sigma=1.2, flags=cv2.OPTFLOW_USE_INITIAL_FLOW, )
    #  , cv2.cv.OPTFLOW_USE_INITIAL_FLOW
    return flow


def hta_algorithm(image_arrays, max_iter=10, video=False):
    """Performs iterative registration for a set of frames.

    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    convergence_threshold = 0.01
    greyscale_frames = [image.mean(axis=2) for image in image_arrays]
    color_frames = image_arrays
    num_frames = len(greyscale_frames)
    temporal_mean = sum(greyscale_frames) / num_frames
    h, w = temporal_mean.shape[:2]

    means = []
    means.append(np.mean(np.stack(color_frames, axis=0), axis=0))

    for iter in range(max_iter):
        shift_maps = []
        shift_maps.append(np.zeros((h, w, 2)))
        for i in range(num_frames-1):
            shift_maps.append(optical_flow(greyscale_frames[i+1], greyscale_frames[0]))

        centroid_shift_map = np.mean(np.stack(shift_maps, axis=0), axis=0)

        for i in range(num_frames):
            # Calculate indices and ensure they stay within array bounds
            indx, indy = np.meshgrid(range(w), range(h), sparse=False, indexing='xy')
            indx = np.fmax(np.fmin(np.rint(indx + centroid_shift_map[:,:,0]), w-1), 0)
            indy = np.fmax(np.fmin(np.rint(indy + centroid_shift_map[:,:,1]), h-1), 0)
            corrected_shift_map = -centroid_shift_map + shift_maps[i][indy.astype(int),indx.astype(int),:]

            greyscale_frames[i] = warp_flow(greyscale_frames[i], corrected_shift_map)
            color_frames[i] = warp_flow(color_frames[i], corrected_shift_map)

        new_mean = sum(greyscale_frames) / num_frames
        diffs = mean_of_differences(temporal_mean, new_mean)
        temporal_mean = new_mean
        means.append(np.mean(np.stack(color_frames, axis=0), axis=0))

        if sum(diffs) < convergence_threshold:
            break

    if video:
        return color_frames

    return means


def oreifej_algorithm(image_arrays, max_iter=10, low_rank=False, video=False):
    """Performs iterative registration for a set of frames.
    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
        low_rank: eliminate sparse noise by rank minimization
        video: return video instead of mean image
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    convergence_threshold = 0.01
    greyscale_frames = [image.mean(axis=2) for image in image_arrays]
    color_frames = image_arrays
    num_frames = len(greyscale_frames)
    temporal_mean = sum(greyscale_frames) / num_frames

    means = []
    means.append(np.mean(np.stack(color_frames, axis=0), axis=0))

    for iter in range(max_iter):
        # Calculate shift maps on-demand to conserve memory
        shift_maps = (optical_flow(frame, temporal_mean) for frame in greyscale_frames)

        for i in range(num_frames):
            pixel_shifts = shift_maps.next()
            greyscale_frames[i] = warp_flow(greyscale_frames[i], pixel_shifts)
            color_frames[i] = warp_flow(color_frames[i], pixel_shifts)

        new_mean = sum(greyscale_frames) / num_frames
        diffs = mean_of_differences(temporal_mean, new_mean)
        temporal_mean = new_mean
        means.append(np.mean(np.stack(color_frames, axis=0), axis=0))

        if sum(diffs) < convergence_threshold:
            break
            
    if video:
        return color_frames

    if low_rank:
        h, w = temporal_mean.shape[:2]
        weight = (h * w) ** -0.5
        component_means = []
        for i in range(3):
            image_vectors = [img[:,:,i].flatten() for img in color_frames]
            frame_matrix = np.array(image_vectors).T
            low_rank_matrix, noise = robust_pca(frame_matrix, alpha=weight)
            component_means.append(low_rank_matrix.mean(axis=1).reshape(h, w))

        means.append(np.stack(component_means, axis=-1))

    return means


def robust_pca(X, alpha=0.01, tol=1e-4, max_iter=50, verbose=False):
    """Sparse error elimination through robust principal component analysis.

    Args:
        X: the matrix to optimize
        alpha: tuning parameter tuning parameter
        tol: convergence tolerance
        max_iter: iteration limit
        verbose: print iteration count when finished

    Returns:
        tuple containing low-rank approximation matrix and sparse noise matrix

    This method is based on inexact augmented lagrange multipliers.
    Original code by Kyle Kastner, http://kastnerkyle.github.io/posts/robust-matrix-decomposition/
    """
    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / alpha
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape, dtype=np.float32)
    E = np.zeros(Y.shape, dtype=np.float32)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    for iter in range(max_iter):
        Eraw = X - A + (1 / mu) * Y
        E = np.maximum(Eraw - alpha / mu, 0) + np.minimum(Eraw + alpha / mu, 0)
        U, S, V = svd(X - E + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        A = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        Z = X - A - E
        Y += mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        if (norm(Z, 'fro') / dnorm) < tol:
            break
    if verbose:
        print("Finished at iteration %d" % (iter+1))
    return A, E


def fluidy(source, filename, max_iter, hta, lowrank, save_each_stage, video):
    """Helper function for accessing image processing algorithm from the command line.

    Args:
        source: name of the video file
        filename: name of target file. Note that multiple files are created!
        max_iter: iteration limit
        hta: flag for enabling the HTA algorithm. If false, Oreifej algorithm is used.
        save_each_stage: flag to write the result of each iteration to disk.

    Returns:
        Nothing, writes output to disk.
    """
    frames = load_frames(source)
    
    if not frames:
        print("No file '{}' found!".format(source))
        return

    if hta:
        imgs = hta_algorithm(frames, max_iter, video)
    else:
        imgs = oreifej_algorithm(frames, max_iter, lowrank, video)

    if video:
        save_video(imgs, filename)
    else:
        if save_each_stage:
            name, ext = os.path.splitext(filename)
            for i in range(len(imgs)):
                save_image(imgs[i], name + str(i) + ext)
        else:
            save_image(imgs[-1], filename)



usage = """
Usage: fluidy.py [args] <source_file> <result_file>

Supported arguments:
--hta: Flag for enabling the HTA algorithm.
-i --iter: Maximum number of iterations.
-l --lowrank: Enable rank minimization in Oreifej algorithm.
-p --progress: Print progress from each iteration
-v --video: Output video instead of still image
"""

if __name__ == "__main__":
    import argparse
    import sys

    if len(sys.argv) < 2 or ("-h", "--help") in sys.argv:
        sys.exit(usage)

    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("dst")
    parser.add_argument("-i", "--iter", type=int, dest="max_iter", default=10)
    parser.add_argument("-l", "--lowrank", action="store_true", dest="lowrank")
    parser.add_argument("-p", "--progress", action="store_true", dest="progress")
    parser.add_argument("-v", "--video", action="store_true", dest="video")
    parser.add_argument("--hta", action="store_true", dest="hta")
    args = parser.parse_args()

    fluidy(args.src, args.dst, args.max_iter, args.hta, 
           args.lowrank, args.progress, args.video)
