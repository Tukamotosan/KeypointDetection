# coding: utf-8

import cv2
import numpy as np

def main():
    path_src = "../images/lena_std.tif"

    # load image
    img_src = cv2.imread(path_src, 1)

    # convert from RGB to grayscale
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    cv2.imshow("Image", img_gry)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
