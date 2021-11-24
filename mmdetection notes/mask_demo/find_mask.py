import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def findMask(train_list,test_list,val_list):

    mask_lists = [train_list,test_list,val_list]

    for i, mask_list in enumerate(mask_lists):

        if i == 0:
            dir = "train/"
        elif i == 1:
            dir = "test/"
        elif i == 2:
            dir = "val/"

        for key, value in mask_list.items():
            # mask_sels.append(value)
            # mask_sel = mask_sels[i]
            mask_sel = value
            mask_sel = cv.cvtColor(mask_sel, cv.COLOR_BGR2GRAY)
            contours, hierarchy = cv.findContours(mask_sel, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
            image_name = key
            kernel = np.ones((3, 3), np.uint8)
            if len(contours)>1:
                for i in range(1,len(contours)):
                    #if contours[0].shape[0] / contours[i].shape[0] < 6 or contours[0].shape[0] / contours[i].shape[0] == 6:
                    cv.fillPoly(mask_sel, [contours[i]], 255)
                    mask_sel = cv.erode(mask_sel, kernel, iterations=1)
                cv.imwrite(os.path.join("/home/data/findmask2/"+dir, image_name),mask_sel)

            else:
                mask_sel = cv.erode(mask_sel, kernel, iterations=1)
                cv.imwrite(os.path.join("/home/data/findmask2/"+dir, image_name), mask_sel)


def walkFile(files):

    if not isinstance(files,list):
        dirs_list = []
        dir_flag = True
        file_flag = False
    else:
        train_list = []
        test_list = []
        val_list = []
        dir_flag = False
        file_flag = True

    if file_flag:
        for file in files:
            for root, dirs, filename in os.walk(file):
                # root 表示当前正在访问的文件夹路径
                # dirs 表示该文件夹下的子目录名list
                # files 表示该文件夹下的文件list

                # 遍历文件
                if root == "/home/data/masks2/train":
                    for f in filename:
                        train_list.append(os.path.join(root, f))
                elif root == "/home/data/masks2/test":
                    for f in filename:
                        test_list.append(os.path.join(root, f))
                else:
                    for f in filename:
                        val_list.append(os.path.join(root, f))

        return train_list,test_list,val_list

    if dir_flag:
        for root, dirs, filename in os.walk(files):
        # 遍历所有的文件夹
            for d in dirs:
                dirs_list.append(os.path.join(root, d))
            return dirs_list


def corrosion_expansion(train_list,test_list,val_list):

    train_corrosion_expansion = {}
    test_corrosion_expansion = {}
    val_corrosion_expansion = {}

    for train in train_list:
        name = train.split("/")[-1]
        img = cv.imread(train)
        kernel = np.ones((3, 3), np.uint8)

        dilation = cv.dilate(img, kernel, iterations=1)
        train_corrosion_expansion[name] = dilation

    for test in test_list:
        name = test.split("/")[-1]
        img = cv.imread(test)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv.dilate(img, kernel, iterations=1)
        test_corrosion_expansion[name] = dilation

    for val in val_list:
        name = val.split("/")[-1]
        img = cv.imread(val)
        kernel = np.ones((3, 3), np.uint8)
        dilation = cv.dilate(img, kernel, iterations=1)
        val_corrosion_expansion[name] = dilation

    return train_corrosion_expansion,  test_corrosion_expansion, val_corrosion_expansion



if __name__ == "__main__":
    path = "/home/data/masks2/"
    dirs_list = walkFile(path)
    train_list,test_list,val_list = walkFile(dirs_list)
    train_list,test_list,val_list = corrosion_expansion(train_list,test_list,val_list)

    findMask(train_list,test_list,val_list)
