"""
Image processing algorithm for seeing through water.

This algorithm reduces distortions in a video caused by water surface and
produce a clear image of the plane below the water surface.
"""

from numpy.linalg import norm, svd

import cv2
import numpy as np


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


def oreifej_algorithm(image_arrays, max_iter=10, low_rank=True):
    """Performs iterative registration for a set of frames.

    Args:
        image_arrays: a list of 3D numpy arrays containing the video frames
        max_iter: Iteration limit
        low_rank: eliminate sparse noise by rank minimization
    Returns:
        a list of numpy arrays containing the mean of each iteration
    """
    # TODO: Blur the frames for a better result
    convergence_threshold = 0.01
    frames = [image.mean(axis=2) for image in image_arrays]
    color_frames = image_arrays
    num_frames = len(frames)
    temporal_mean = sum(frames) / num_frames

    means = []
    means.append(np.mean(np.stack(color_frames,axis = 0),axis = 0))

    for iter in range(max_iter):
        shift_maps = [optical_flow(frames[i], temporal_mean) for i in range(len(frames))]
        dewarped_frames = [warp_flow(frames[i], shift_maps[i]) for i in range(len(frames))]
        color_frames = [warp_flow(color_frames[i], shift_maps[i]) for i in range(len(color_frames))]

        new_mean = sum(dewarped_frames) / num_frames
        diffs = mean_of_differences(temporal_mean, new_mean)
        temporal_mean = new_mean
        means.append(np.mean(np.stack(color_frames,axis = 0),axis = 0))
        frames = dewarped_frames
        
        if diffs[0] < convergence_threshold:
            break

    if low_rank:
        h, w = temporal_mean.shape[:2]
        weight = (h * w) ** -0.5
        component_means = []
        for i in range(3):
            image_vectors = [img[:,:,i].flatten() for img in color_frames]
            frame_matrix = np.array(image_vectors).T
            low_rank_matrix, noise = robust_pca(frame_matrix, alpha=weight)
            component_means.append(low_rank_matrix.mean(axis=1).reshape(h, w))
        
        means.append(np.stack(component_means,axis = -1))

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
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
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


#   Testing	
if __name__ == "__main__":
    frames = load_frames('expdata_middle.avi')
    imgs = oreifej_algorithm(frames[0:99])
    for i in range(len(imgs)): save_image(imgs[i], "result5{}.jpg".format(i))
