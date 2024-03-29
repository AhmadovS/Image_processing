# CSC320 Winter 2017
# Assignment 4
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
# import the heapq package
from heapq import heappush, heappushpop, nlargest, heappop
# see below for a brief comment on the use of tiebreakers in python heaps
from itertools import count
_tiebreaker = count()

from copy import deepcopy as copy

# basic numpy configuration

# set random seed
np.random.seed(seed=131)
# ignore division by zero warning
np.seterr(divide='ignore', invalid='ignore')


# This function implements the basic loop of the Generalized PatchMatch
# algorithm, as explained in Section 3.2 of the PatchMatch paper and Section 3
# of the Generalized PatchMatch paper.
#
# The function takes k NNFs as input, represented as a 2D array of heaps and an
# associated 2D array of dictionaries. It then performs propagation and random search
# as in the original PatchMatch algorithm, and returns an updated 2D array of heaps
# and dictionaries
#
# The function takes several input arguments:
#     - source_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the source image,
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
#     - target_patches:      *** Identical to A3 ***
#                            The matrix holding the patches of the target image.
#     - f_heap:              For an NxM source image, this is an NxM array of heaps. See the
#                            helper functions below for detailed specs for this data structure.
#     - f_coord_dictionary:  For an NxM source image, this is an NxM array of dictionaries. See the
#                            helper functions below for detailed specs for this data structure.
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
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            you can pass them to/from your function using this argument

# Return arguments:
#     - global_vars:         (optional) if you want your function to use any global variables,
#                            return them in this argument and they will be stored in the
#                            PatchMatch data structure
#     NOTE: the variables f_heap and f_coord_dictionary are modified in situ so they are not
#           explicitly returned as arguments to the function


def propagation_and_random_search_k(source_patches, target_patches,
                                    f_heap,
                                    f_coord_dictionary,
                                    alpha, w,
                                    propagation_enabled, random_enabled,
                                    odd_iteration,
                                    global_vars
                                    ):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    shp = source_patches.shape
    ## Get nnf displacements and distances
    f, D = NNF_heap_to_NNF_matrix(f_heap)

    ## Find k (number of nnf displacements per point)
    tpl_len = len(f_heap[0][0])

    for x in range(shp[0]):
        for y in range(shp[1]):

            ## patch with all color intensities for x,y
            src_patch = source_patches[x, y, :, :]

            ##Propation
            if propagation_enabled:

                ##Odd iterations
                ## Calculate for each neighbor

                for k in range(tpl_len):
                    o = f[k][x][y]
                    idxs =[o]
                    if odd_iteration:
                        ## Boundary check
                        if x != 0:
                            idxs.append(f[k][x-1][y])
                        if y != 0:
                            idxs.append(f[k][x][y-1])
                        idxs = np.vstack(idxs)

                    ##Even iterations
                    else:
                        ## Boundary check
                        if x < shp[0]-1:
                            idxs.append(f[k][x+1][y])
                        if y < shp[1] -1 :
                            idxs.append(f[k][x][y+1])
                        idxs = np.vstack(idxs)

                    idxs = idxs.astype(int)

                    ## Calculate targets indices
                    targets = idxs + [x, y]

                    ## ignore target indices those are out of bounds
                    for tar in targets:
                        if 0<=tar[0] < shp[0] and 0<=tar[1] < shp[1]:

                            ## Get the target patch
                            all_targets = target_patches[tar[0], tar[1], :, :]

                            ## Find difference between target patch and source patch

                            res = (all_targets - src_patch) ** 2

                            ## Ignore all the NaN values
                            non_nans = res[~np.isnan(res)]
                            ## Calculate the distance
                            # result = np.sqrt(np.mean(non_nans))
                            result = np.mean(non_nans)
                            ## Create new entry for heap
                            tmp_min = result
                            tmp_disp = [tar[0] - x, tar[1] - y]
                            tmp_tpl = -tmp_min, next(_tiebreaker), tmp_disp
                            ## If better match found
                            if tmp_min < D[k,x,y]:
                                ## And displacment hasnt been added before
                                if tuple(tmp_disp) not in f_coord_dictionary[x][y]:
                                    ## Update the values
                                    D[k,x,y] = tmp_min
                                    heappushpop(f_heap[x][y], tmp_tpl)
                                    f_coord_dictionary[x][y][tuple(tmp_disp)] = 0
                                    f[k,x,y] = tmp_disp
            if random_enabled:

                ## Calculate the number of iterations
                ## w*alpha^i <=1 => log (w*alpha^i) <= 0 (log1)
                ## log w + i*log alpha <= 1
                ## i <= -log w/ log alpha
                max_iters = np.int(-np.log(w) / np.log(alpha))
                ## Do calculation for each k
                for k in range(tpl_len):
                    ## Old displacement value
                    old_v = f[k][x][y]
                    ## Generate R for each
                    r = np.random.uniform(low=-1,high= 1, size=(2, max_iters+1))

                    ## Calculate the offsets
                    off_func = lambda i:  r[:,i]*w * (alpha ** i)
                    offsets =  map(off_func, range(max_iters+1))
                    offsets =  offsets +old_v
                    offsets = np.asarray(offsets).astype(int)
                    ## Calculate the indices of all target patches
                    targets = offsets +  [x,y]

                    ## Ignore the invalid indices
                    for tar in targets:
                        if 0 <= tar[0] < shp[0] and 0 <= tar[1] < shp[1]:
                            ## Get target patch
                            all_targets = target_patches[tar[0], tar[1],:,:]
                            ## Calculate the difference between target and source patch
                            res = (all_targets - src_patch)**2
                            ## Ignore NaNs
                            non_nans = res[~np.isnan(res)]

                            ## Calculate the similarity between target patch and source patch
                            result = np.mean(non_nans)
                            # result = np.sqrt(np.mean(non_nans))

                            ## Create a new entry for heap
                            tmp_min = result
                            tmp_disp = [tar[0]-x,tar[1]-y]
                            tmp_tpl = -tmp_min, next(_tiebreaker), tmp_disp

                            ## If better patch found update the heap
                            if tmp_min < D[k,x,y]:
                                if tuple(tmp_disp) not in f_coord_dictionary[x][y]:

                                    heappushpop(f_heap[x][y], tmp_tpl)
                                    D[k,x,y] = tmp_min
                                    f[k, x, y] = tmp_disp
                                    f_coord_dictionary[x][y][tuple(tmp_disp)] = 0
    #############################################

    return global_vars


# This function builds a 2D heap data structure to represent the k nearest-neighbour
# fields supplied as input to the function.
#
# The function takes three input arguments:
#     - source_patches:      The matrix holding the patches of the source image (see above)
#     - target_patches:      The matrix holding the patches of the target image (see above)
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds k NNFs. Specifically,
#                            f_k[i] is the i-th NNF and has dimension NxMx2 for an NxM image.
#                            There is NO requirement that f_k[i] corresponds to the i-th best NNF,
#                            i.e., f_k is simply assumed to be a matrix of vector fields.
#
# The function should return the following two data structures:
#     - f_heap:              A 2D array of heaps. For an NxM image, this array is represented as follows:
#                               * f_heap is a list of length N, one per image row
#                               * f_heap[i] is a list of length M, one per pixel in row i
#                               * f_heap[i][j] is the heap of pixel (i,j)
#                            The heap f_heap[i][j] should contain exactly k tuples, one for each
#                            of the 2D displacements f_k[0][i][j],...,f_k[k-1][i][j]
#
#                            Each tuple has the format: (priority, counter, displacement)
#                            where
#                                * priority is the value according to which the tuple will be ordered
#                                  in the heapq data structure
#                                * displacement is equal to one of the 2D vectors
#                                  f_k[0][i][j],...,f_k[k-1][i][j]
#                                * counter is a unique integer that is assigned to each tuple for
#                                  tie-breaking purposes (ie. in case there are two tuples with
#                                  identical priority in the heap)
#     - f_coord_dictionary:  A 2D array of dictionaries, represented as a list of lists of dictionaries.
#                            Specifically, f_coord_dictionary[i][j] should contain a dictionary
#                            entry for each displacement vector (x,y) contained in the heap f_heap[i][j]
#
# NOTE: This function should NOT check for duplicate entries or out-of-bounds vectors
# in the heap: it is assumed that the heap returned by this function contains EXACTLY k tuples
# per pixel, some of which MAY be duplicates or may point outside the image borders

def NNF_matrix_to_NNF_heap(source_patches, target_patches, f_k):

    ## Init the variables
    shp = f_k.shape
    f_heap = []
    f_coord_dictionary = []

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    for x in range(shp[1]):
        ## Add new list for each row
        f_heap.append([])
        f_coord_dictionary.append([])
        for y in range(shp[2]):

            ## patch with all color intensities for x,y
            src_patch = source_patches[x, y, :, :]
            ## Displacement value
            disp = f_k[:,x,y,:]
            ## Heap and dict
            h =[]
            dct = {}
            for i in range(disp.shape[0]):
                dx, dy = disp[i,:]
                ## If invalid entry add inf
                if x + dx >= shp[1] or y+dy>=shp[2]:
                    tpl = float('-inf'), next(_tiebreaker), disp[i,:]
                    heappush(h,tpl)

                else:
                    ## Else similarity between pacthes
                    trgt_ptch = target_patches[x+dx,y+dy,:,:]
                    res = (trgt_ptch - src_patch)**2
                    n_nan = res[~np.isnan(res)]

                    # sim = -np.sqrt(np.mean(n_nan))
                    sim = -np.mean(n_nan)

                    tpl = sim, next(_tiebreaker), disp[i, :]
                    heappush(h, tpl)

                dct[tuple([dx,dy])] = 0

            f_heap[x].append(h)
            f_coord_dictionary[x].append(dct)
    #############################################

    return f_heap, f_coord_dictionary


# Given a 2D array of heaps given as input, this function creates a kxNxMx2
# matrix of nearest-neighbour fields
#
# The function takes only one input argument:
#     - f_heap:              A 2D array of heaps as described above. It is assumed that
#                            the heap of every pixel has exactly k elements.
# and has two return arguments
#     - f_k:                 A numpy array of dimensions kxNxMx2 that holds the k NNFs represented by the heap.
#                            Specifically, f_k[i] should be the NNF that contains the i-th best
#                            displacement vector for all pixels. Ie. f_k[0] is the best NNF,
#                            f_k[1] is the 2nd-best NNF, f_k[2] is the 3rd-best, etc.
#     - D_k:                 A numpy array of dimensions kxNxM whose element D_k[i][r][c] is the patch distance
#                            corresponding to the displacement f_k[i][r][c]
#

def NNF_heap_to_NNF_matrix(f_heap):

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    k = len(f_heap[0][0])
    shp = [len(f_heap), len(f_heap[0])]

    ## Init arrays
    D_k = np.zeros((k,shp[0],shp[1]))
    f_k = np.zeros((k, shp[0], shp[1],2))

    for x in range(shp[0]):
        for y in range(shp[1]):

            ## Get the largest values
            arrs = nlargest(k, f_heap[x][y])
            for i in range(len(arrs)):
                ## Update the values
                f_k[i,x,y] = arrs[i][2]
                D_k[i,x,y] = -arrs[i][0]

    #############################################
    f_k = f_k.astype(int)

    return f_k, D_k


def nlm(target, f_heap, h):


    # this is a dummy statement to return the image given as input
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    #############################################

    ## Init denoised
    denoised = np.zeros(target.shape)

    ## Get values to calculate the patch in target
    g = make_coordinates_matrix(target.shape)
    ## NNF and similarity vectors
    f, D = NNF_heap_to_NNF_matrix(f_heap)

    ## Calculate the weights
    w = np.exp(-1*D/h**2)
    ## Normalize the weights
    weights = w/np.sum(w, axis=0)

    for i in range(f.shape[0]):

        targets = target[g[:,:,0]+f[i,:,:,0],g[:,:,1]+f[i,:,:,1],:]
        denoised+= targets*weights[i,:,:, np.newaxis]

    #############################################

    return denoised




#############################################
###  PLACE ADDITIONAL HELPER ROUTINES, IF ###
###  ANY, BETWEEN THESE LINES             ###
#############################################


#############################################



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


    ################################################
    ###  PLACE YOUR A3 CODE BETWEEN THESE LINES  ###
    ################################################

    ## Get the coordinates
    g = make_coordinates_matrix(target.shape)
    ## Add offsets to coordinates
    g = g.astype(int)
    f = f.astype(int)
    g += f
    ## Get the source
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
