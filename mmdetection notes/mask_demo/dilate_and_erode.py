import os
import cv2 as cv
import numpy as np


def erode_demo(image):

    kernel = np.ones((3, 3), np.uint8)
    dst = cv.erode(image, kernel, iterations=1)
    return dst


def dilate_demo(image):

    kernel = np.ones((9, 9), np.uint8)
    dst = cv.dilate(image, kernel, iterations=1)
    return dst


path = "C:\\Users\\Administrator\\Desktop\\mask"
save_path = "C:\\Users\\Administrator\\Desktop\\mask2\\"
for file in os.listdir(path):
    if "mask" in file:
        file_path = os.path.join(path,file)
        img = cv.imread(file_path)
        img = dilate_demo(img)
        mask_sel = img
        mask_sel = cv.cvtColor(mask_sel, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(mask_sel, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        kernel = np.ones((3, 3), np.uint8)
        # img = dilate_demo(img)
        # img = erode_demo(img)
        if len(contours) > 1:
            for i in range(1, len(contours)):
                # if contours[0].shape[0] / contours[i].shape[0] < 6 or contours[0].shape[0] / contours[i].shape[0] == 6:
                cv.fillPoly(mask_sel, [contours[i]], 255)
                mask_sel = cv.erode(mask_sel, kernel, iterations=1)
            cv.imwrite(save_path + file, mask_sel)

        else:
            mask_sel = cv.erode(mask_sel, kernel, iterations=1)
            cv.imwrite(save_path + file, mask_sel)