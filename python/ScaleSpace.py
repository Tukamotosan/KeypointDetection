# coding: utf-8

import cv2
import numpy as np
import pprint

def f_s(t, T):
    """
    function S in Algorithm 2
    :param t:
    :param T:
    :return:
    """
    t1 = t % (2*T)
    t2 = (2*T - 1 - t) % (2*T)
    return min([t1, t2])

def f_u_bar(k, l, M, N, img):
    """
    function u^bar in Algorithm 2
    :param k:
    :param l:
    :param M:
    :param N:
    :param img:
    :return:
    """
    sMk = f_s(k, M)
    sNl = f_s(l, N)
    return img[sNl, sMk][0]

def bilinear_interpolation(img_src, delta_dash):
    """
    Algorithm 2 Bilinear interpolation of an image
    :param img_src:
    :param delta_dash:
    :return:
    """
    N,M = img_src.shape[:2]
    Ndash,Mdash = float(int(N/delta_dash)), float(int(M/delta_dash))
    img = np.zeros((int(Ndash), int(Mdash), 1), np.uint8)

    for m_dash in range(int(Mdash)):
        for n_dash in range(int(Ndash)):
            x, y = delta_dash*m_dash, delta_dash*n_dash

            img[]



    return None

def main():
    path_src = "../images/lena_std.tif"

    # load image
    img_src = cv2.imread(path_src, 1)

    # convert from RGB to grayscale
    img_gry = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    # bilinear interpolation
    delta_dash = 0.8
    img_org = bilinear_interpolation(img_gry, delta_dash)

    # cv2.imshow("Image", img_gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
