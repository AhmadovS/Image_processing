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

    if best_D is None:
        best_D = np.zeros((shp[0],shp[1],1))*np.float('inf')

    for x in range(shp[0]):
        for y in range(shp[1]):
            src_patch = source_patches[x, y, :, :]
            n_nan = len(src_patch[~np.isnan(src_patch)])
            src_patch[np.isnan(src_patch)] = 0

            ##Propation
            if not propagation_enabled:
                ##Odd iterations
                if odd_iteration:
                    ##Left patch
                    if x + f[x-1, y][0] < shp[0] and y + f[x-1, y][1] <shp[1]:
                        trgt_patch = target_patches[x + f[x-1, y][0], y + f[x-1, y][1], :, :]
                        trgt_patch[np.isnan(trgt_patch)] = 0
                        tmp_D = np.sqrt(np.sum((trgt_patch - src_patch) ** 2) / n_nan)

                        if tmp_D< best_D[x,y]:
                            best_D[x,y] = tmp_D
                            new_f[x,y] = f[x-1,y]

                    ##Up patch
                    if x + f[x, y-1][0] < shp[0] and y + f[x, y-1][1] < shp[1]:
                        trgt_patch = target_patches[x + f[x, y-1][0], y + f[x, y-1][1], :, :]
                        trgt_patch[np.isnan(trgt_patch)] = 0
                        tmp_D = np.sqrt(np.sum((trgt_patch - src_patch) ** 2) / n_nan)
                        if tmp_D< best_D[x,y]:
                            best_D[x,y] = tmp_D
                            new_f[x,y] = f[x,y-1]

                ##Even iterations
                else:

                    if x + 1<shp[0] and  x + f[x + 1, y][0] < shp[0] and y + f[x + 1, y][1]<shp[1]:
                        ##RIght patch
                        trgt_patch = target_patches[x + f[x + 1, y][0], y + f[x + 1, y][1], :, :]
                        trgt_patch[np.isnan(trgt_patch)] = 0
                        tmp_D = np.sqrt(np.sum((trgt_patch - src_patch) ** 2) / n_nan)

                        if tmp_D< best_D[x,y]:
                            best_D[x,y] = tmp_D
                            new_f[x,y] = f[x+1,y]

                    if y + 1 <shp[1] and  x + f[x, y+1][0] < shp[0] and y + f[x , y+1][1]<shp[1]:
                        ##Down patch
                        trgt_patch = target_patches[x + f[x, y+1][0], y + f[x , y+1][1], :, :]
                        trgt_patch[np.isnan(trgt_patch)] = 0
                        tmp_D = np.sqrt(np.sum((trgt_patch - src_patch) ** 2) / n_nan)

                        if tmp_D < best_D[x, y]:
                            best_D[x, y] = tmp_D
                            new_f[x, y] = f[x , y+1]

                #     idxs = [new_f[x,y,:], ]
                #     if x+1<shp[0] and x+ new_f[x+1,y,:][0] < shp[0]:
                #         idxs.append(new_f[x+1,y,:])
                #     else:
                #         idxs.append(np.NaN)
                #     if y + 1<shp[1] and y+ new_f[x,y+1,:][1] < shp[1]:
                #         idxs.append(new_f[x,y + 1,:])
                #     else:
                #         idxs.append(np.NaN)
                #
                # res = []
                #
                # ##Troubleshooting and find min
                # for id in idxs:
                #     if np.any(np.isnan(id)) or abs(x + id[0]) >= shp[0] or abs(y + id[1]) >= shp[1]:
                #         res.append(np.float('inf'))
                #     else:
                #         trgt_patch = target_patches[x+id[0],y+id[1],:,:]
                #         trgt_patch[np.isnan(trgt_patch)] = 0
                #         res.append(np.sqrt(np.sum((trgt_patch-src_patch)**2)/n_nan))
                #
                # tmp_D = np.min(res)
                # if np.isnan(best_D[x,y]) or tmp_D < best_D[x,y]:
                #     best_D[x,y,:] = tmp_D
                #
                # new_f[x,y,:] = idxs[np.argmin(res)]

            if not random_enabled:
                old_v = f[x, y, :]
                i = 0
                offset = w
                while offset > 1:
                    r = np.random.choice([-1,1], size=2)

                    u = old_v + np.round(r*offset)
                    # u_x = np.round(old_v[0] + r[0]*offset)
                    # u_y = np.round(old_v[1] + r[1] * offset)

                    # u_x = int(np.asscalar(u_x))
                    # u_y = int(np.asscalar(u_y))
                    ux = x + int(u[0])
                    uy = y + int(u[1])
                    #
                    # if abs(ux)>=shp[0]:
                    #     u[0] = int(u[0]-shp[0]*(ux/shp[0]))
                    #     ux = x + int(u[0])
                    # if abs(uy)>=shp[1]:
                    #     u[1] = int(u[1] - shp[1] * (uy / shp[1]))
                    #     uy = y + int(u[1])

                    if abs(ux) < shp[0] and abs(uy) < shp[1]:
                        trgt_patch = target_patches[ux, uy, :, :]
                        trgt_patch[np.isnan(trgt_patch)] = 0

                        tmp_D = np.sqrt(np.sum((trgt_patch-src_patch)**2)/n_nan)
                        if np.isnan(best_D[x,y]) or tmp_D < best_D[x,y]:

                            best_D[x, y,:] = tmp_D
                            new_f[x, y, :] = u
                    i+=1
                    offset = w*(alpha**i)


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
    g+=f

    rec_source = target[g[:,:,0],g[:,:,1],:]
    
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
