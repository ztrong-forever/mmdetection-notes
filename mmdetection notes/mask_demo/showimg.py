import cv2
import os


path = "C:\\Users\\Administrator\\Desktop\\mask"
save_path = "C:\\Users\\Administrator\\Desktop\\mask2\\"
for file in os.listdir(path):
    if "mask" in file:
        file_path = os.path.join(path, file)
        img = cv2.imread(file_path)
        median = cv2.medianBlur(img, 21)
        cv2.imwrite(save_path + file, median)


