# CSC320 Winter 2017
# Assignment 3
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY PROHIBITED. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# import basic packages
import numpy as np

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the PatchMatch
# algorithm, as explained in Section 3.2 of the paper.
# The function takes an NNF f as input, performs propagation and random search,
# and returns an updated NNF.
#
# The function takes several input arguments:
#     - source_patches:      The matrix holding the patches of the source image,
#                            as computed by the make_patch_matrix() function. For an
#                            NxM source image and patches of width P, the matrix has
#                            dimensions NxMxCx(P^2) where C is the number of color channels
#                            and P^2 is the total number of pixels in the patch. The
#                            make_patch_matrix() is defined below and is called by the
#                            initialize_algorithm() method of the PatchMatch class. For
#                            your purposes, you may assume that source_patches[i,j,c,:]
#                            gives you the list of intensities for color channel c of
#                            all pixels in the patch centered at pixel [i,j]. Note that patches
#                            that go beyond the image border will contain NaN values for
#                            all patch pixels that fall outside the source image.
#     - target_patches:      The matrix holding the patches of the target image.
#     - f:                   The current nearest-neighbour field
#     - alpha, w:            Algorithm parameters, as explained in Section 3 and Eq.(1)
#     - propagation_enabled: If true, propagation should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step
#     - random_enabled:      If true, random search should be performed.
#                            Use this flag for debugging purposes, to see how your
#                            algorithm performs with (or without) this step.
#     - odd_iteration:       True if and only if this is an odd-numbered iteration.
#                            As explained in Section 3.2 of the paper, the algorithm
#                            behaves differently in odd and even iterations and this
#                            parameter controls this behavior.
#     - best_D:              And NxM matrix whose element [i,j] is the similarity score between
#                            patch [i,j] in the source and its best-matching patch in the
#                            target. Use this matrix to check if you have found a better
#                            match to [i,j] in the current PatchMatch iteration
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - new_f:               The updated NNF
#     - best_D:              The updated similarity scores for the best-matching patches in the
#                            target
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure


def propagation_and_random_search(source_patches, target_patches,
                                  f, alpha, w,
                                  propagation_enabled, random_enabled,
                                  odd_iteration, best_D=None,
                                  global_vars=None
                                  ):
    new_f = f.copy()

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################
    shp = source_patches.shape

    ## If it is first iteration initilaize Best_D to be inf in all points
    if best_D is None:
        best_D = np.ones((shp[0], shp[1], 1)) * np.inf

    for x in range(shp[0]):
        for y in range(shp[1]):

            ## patch with all color intensities for x,y
            src_patch = source_patches[x, y, :, :]

            ##Propation
            if not propagation_enabled:
                ##Odd iterations
                if odd_iteration:
                    idxs = [new_f[x, y]]
                    ## Boundary check
                    if x != 0:
                        idxs.append(new_f[x - 1, y])
                    if y != 0:
                        idxs.append(new_f[x, y-1])
                    idxs = np.vstack(idxs)

                ##Even iterations
                else:
                    idxs = [new_f[x, y]]
                    ## Boundary check
                    if x!=shp[0]-1:
                        idxs.append(new_f[x+1, y, :])
                    if y!=shp[1]-1:
                        idxs.append(new_f[x, y+1, :])
                    idxs = np.vstack(idxs)

                ## Calculate targets indices
                targets = idxs + [x, y]

                ## Find all invalid indices
                x_bound = (abs(targets[:, 0]) >= shp[0])
                y_bound = (abs(targets[:, 1]) >= shp[1])

                ## Clamp the invalid indices into bound
                idxs[x_bound, 0] -= shp[0] * targets[x_bound, 0] // shp[0]
                idxs[y_bound, 1] -= shp[1] * targets[y_bound, 1] // shp[1]

                ## Get target patches
                all_targets = target_patches[idxs[:, 0] + x, idxs[:, 1] + y]

                ## Find difference between target patches and source patch
                res = (all_targets - src_patch)
                ## Reshape it to make calculation easier
                res = res.reshape(res.shape[0], res.shape[1] * res.shape[2])
                ## Find the all NaNs in results
                nan_index = np.isnan(res)
                ## Boolean array contains True for each non-NaN entry
                n_nan = (~nan_index)

                ## Find the number of valid entries for each patch
                non_nans = np.sum(n_nan, axis=1)

                res[nan_index] = 0
                ## Calculate the similarity. Use RMS
                result = np.sqrt(np.sum(res ** 2, axis=1) / non_nans)
                ## Find the most simirlar patch
                tmp_min = min(result)
                ## If better patch found update the offsets and best_D
                if  tmp_min < best_D[x, y, :]:
                    best_D[x, y, :] = tmp_min
                    new_f[x, y, :] = idxs[np.argmin(result), :]


            if not random_enabled:

                ## Calculate the number of iterations
                ## w*alpha^i <=1 => log (w*alpha^i) <= 0 (log1)
                ## log w + i*log alpha <= 1
                ## i <= -log w/ log alpha
                max_iters = np.int(-np.log(w) / np.log(alpha))+1

                ## Get the old offsets
                old_v = new_f[x, y, :]

                ## Generate R for each
                r = np.random.uniform(low=-1,high= 1, size=(2, max_iters+1))

                ## Calculate the offsets
                off_func = lambda i:  r[:,i]*w * (alpha ** i)
                offsets =  map(off_func, range(max_iters+1))
                offsets = old_v + np.asarray(offsets).astype(int)

                ## Calculate the indices of all target patches
                targets = offsets +  [x,y]

                ## Find invalid indices and clamp them into the bounds
                x_bound = (abs(targets[:,0]) >=shp[0])
                y_bound = (abs(targets[:,1]) >= shp[1])

                offsets[x_bound, 0] -= shp[0] * targets[x_bound, 0] // shp[0]
                offsets[y_bound, 1] -= shp[1] * targets[y_bound, 1] // shp[1]

                ## Get all target patches
                all_targets = target_patches[offsets[:,0]+x, offsets[:,1]+y]
                ## Calculate the difference between targets and source patch
                res = (all_targets - src_patch)
                ## Reshape it to make calculation easier0
                res = res.reshape(res.shape[0], res.shape[1]*res.shape[2])
                ## Find indices of NaNs
                nan_index = np.isnan(res)
                ## The boolean matrix where it is True for each non-NaN
                n_nan = (~nan_index)
                ## Find the number of valid entries for each patch
                non_nans = np.sum(n_nan, axis=1)

                ## Calculate the similarity between target patches and source patch
                res[nan_index] = 0
                result = np.sqrt(np.sum(res**2, axis=1)/non_nans)
                ## Find the most similar patch
                tmp_min = min(result)
                ## If better patch found update the best_D and offsets
                if  tmp_min < best_D[x,y,:]:
                    best_D[x,y,:] = tmp_min
                    new_f[x,y,:] = offsets[np.argmin(result),:]
    #############################################

    return new_f, best_D, global_vars


# This function uses a computed NNF to reconstruct the source image
# using pixels from the target image. The function takes two input
# arguments
#     - target: the target image that was used as input to PatchMatch
#     - f:      the nearest-neighbor field the algorithm computed
# and should return a reconstruction of the source image:
#     - rec_source: an openCV image that has the same shape as the source image
#
# To reconstruct the source, the function copies to pixel (x,y) of the source
# the color of pixel (x,y)+f(x,y) of the target.
#
# The goal of this routine is to demonstrate the quality of the computed NNF f.
# Specifically, if patch (x,y)+f(x,y) in the target image is indeed very similar
# to patch (x,y) in the source, then copying the color of target pixel (x,y)+f(x,y)
# to the source pixel (x,y) should not change the source image appreciably.
# If the NNF is not very high quality, however, the reconstruction of source image
# will not be very good.
#
# You should use matrix/vector operations to avoid looping over pixels,
# as this would be very inefficient

def reconstruct_source_from_target(target, f):
    rec_source = np.zeros(target.shape)
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    g = make_coordinates_matrix(target.shape)
    g += f

    rec_source = target[g[:, :, 0], g[:, :, 1], :]

    #############################################

    return rec_source


# This function takes an NxM image with C color channels and a patch size P
# and returns a matrix of size NxMxCxP^2 that contains, for each pixel [i,j] in
# in the image, the pixels in the patch centered at [i,j].
#
# You should study this function very carefully to understand precisely
# how pixel data are organized, and how patches that extend beyond
# the image border are handled.


def make_patch_matrix(im, patch_size):
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, im.shape[1] + patch_size - 1, im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel. If the
    # original image had NxM pixels, this matrix will have NxMx(patch_size*patch_size)
    # pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


# Generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
# such that g(y,x) = [y,x]
#
# Step is an optional argument used to create a matrix that is step times
# smaller than the full image in each dimension
#
# Pay attention to this function as it shows how to perform these types
# of operations in a vectorized manner, without resorting to loops


def make_coordinates_matrix(im_shape, step=1):
    """
    Return a matrix of size (im_shape[0] x im_shape[1] x 2) such that g(x,y)=[y,x]
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
