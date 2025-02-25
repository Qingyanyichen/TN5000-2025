#####################################################
### DETECTS SALIENT REGIONS FROM SALIENCY MAP #######
########## CODED BY SAI PRAJWAL KOTAMRAJU ###########
#####################################################

from skimage.measure import regionprops
from skimage import measure
import numpy as np
import cv2

def salient_regions_in_salMap(img, area_thresh):
    ###labels = measure.label(img, neighbors=8, background=0)
    labels = measure.label(img, background=0)
    # loop over the unique components
    list_of_sal_regs = []
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        labelMask = np.zeros(img.shape, dtype="uint8")
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
        # if the number of pixels in the component is sufficiently
	    # large, then add it to our mask of "large blobs"
        if numPixels > (img.shape[0]*img.shape[1])*area_thresh:
            properties = regionprops(labelMask)
            for i in range(len(properties)):
                x, y = properties[i].bbox[1], properties[i].bbox[0]
                pt1 = (x, y)
                d1, d2 = properties[i].bbox[3], properties[i].bbox[2]
                delta = (d1, d2)
                list_of_sal_regs.append((x, y, d1, d2))
    return list_of_sal_regs

def find_sal_regions(img, area_thresh=0.005):
    img = cv2.erode(img, None, iterations=4)
    img = cv2.dilate(img, None, iterations=4)
    return salient_regions_in_salMap(img, area_thresh)
