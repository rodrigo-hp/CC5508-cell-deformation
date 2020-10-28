from scipy import ndimage
import numpy as np
import cv2
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max


# receives an image path with tif format and it returns a tuple with the images
def readTif(image_file, flag=cv2.IMREAD_GRAYSCALE):
    mat = cv2.imreadmulti(image_file, flags=flag)
    result = mat[1]
    return result


# receives an image and plots it with matplotlib
def showImage(image):
    plt.imshow(image, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


# receives an array of images and shows the index image with matplotlib
def showImageFromArray(array, index):
    plt.imshow(array[index], cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.show()


# receives two images and its titles to plot them
def showTwoImages(image1, firstTitle, image2, secondTitle):
    plt.subplot(1, 2, 1), plt.imshow(image1, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.xlabel(firstTitle)
    plt.subplot(1, 2, 2), plt.imshow(image2, cmap='gray')
    plt.xticks([]), plt.yticks([])
    plt.xlabel(secondTitle)


# receives an image and applies anisotropic diffusion on it
def anisotropicDiff(image, iterations=20, delta=0.14, kappa=15):
    # convert input image
    im = image.astype('float64')

    # initial condition
    u = im

    # center pixel distances
    dd = np.sqrt(2)

    # 2D finite difference windows
    windows = [
        np.array(
            [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
            [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
        ),
        np.array(
            [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
        ),
        np.array(
            [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
            [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
            [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
        ),
        np.array(
            [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
        ),
        np.array(
            [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
    ]

    for r in range(iterations):
        # approximate gradients
        grads = [ndimage.filters.convolve(u, w) for w in windows]

        # approximate diffusion function
        diff = [1. / (1 + (n / kappa) ** 2) for n in grads]

        # update image
        terms = [diff[i] * grads[i] for i in range(4)]
        terms += [(1 / (dd ** 2)) * diff[i] * grads[i] for i in range(4, 8)]
        u = u + delta * (sum(terms))

    return u


# transforms the elements of an image to UINT8 type
def toUINT8(image):
    image[image < 0] = 0
    image[image > 255] = 255
    image = image.astype(np.uint8, copy=False)
    return image


# receives an image and applies different morphological operations to obtain a valid segmentation,
# returns the filtered image,
def getSegmentation(image):
    segmentation = toUINT8(anisotropicDiff(image, iterations=3))
    ret, binary = cv2.threshold(segmentation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel3, iterations=3)
    closing = cv2.bitwise_not(closing)

    # sure background area
    sure_bg = cv2.dilate(closing, kernel3, iterations=3)

    # Finding distance transformation for local max
    dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
    localMax = peak_local_max(dist_transform, indices=False, min_distance=10, labels=closing)
    maxes = localMax.astype(int)
    # Finding sure foreground area
    sure_fg = cv2.morphologyEx(toUINT8(maxes), cv2.MORPH_DILATE, kernel3, iterations=3)
    sure_fg *= 255
    # we obtain the area between contours
    unknown = cv2.subtract(sure_bg, sure_fg)
    # labeling the contours
    markers = ndimage.label(sure_fg, structure=np.ones((3, 3)))[0]
    markers = markers + 1
    markers[unknown == 255] = 0
    imcolor = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    labels = cv2.watershed(imcolor, markers)
    segmentated = np.zeros(image.shape, dtype="uint8")
    moments = []
    momentAreas = []
    contours = []
    for label in np.unique(labels):
        # if the label is zero, we are examining the 'background'
        # so simply ignore it
        if label == 1 or label == -1:
            continue
        # otherwise, allocate memory for the label region and draw
        # it on the mask
        mask = np.zeros(image.shape, dtype="uint8")
        mask[labels == label] = 255

        # detect contours in the mask and grab the largest one
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c = max(cnts, key=cv2.contourArea)
        # draw a circle enclosing the object
        cv2.drawContours(segmentated, [c], -1, 255, -1)
        contours.append(c)
        M = cv2.moments(c)
        moments.append(M)
        momentAreas.append(M['m00'])

    return moments, momentAreas, contours



def paintCells(image, momentAreas, contours):
    datamatrix = [momentAreas, contours]
    st = sorted(map(list, zip(*datamatrix)), key=lambda x: x[0], reverse=False)
    segmentated = np.zeros(image.shape, dtype="uint8")
    for i in range(len(st)):
        cv2.drawContours(segmentated, [st[i][1]], -1, i, -1)

    return segmentated


# receives an original image and returns a list with the moments of all the detected cells
def getOriginalMoments(image):
    im2, contours, hierarchy = cv2.findContours(image, 1, 2)
    moments = []
    for i in range(len(contours)):
        M = cv2.moments(contours[i])
        moments.append(M)

    return moments
