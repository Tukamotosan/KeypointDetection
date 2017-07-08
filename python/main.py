# coding: utf-8

import cv2
import numpy as np
import pprint
from ScaleSpace import ScaleSpace

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
    return img[int(sNl), int(sMk)]

def bilinear_interpolation(img_src, delta_dash):
    """
    Algorithm 2 Bilinear interpolation of an image
    :param img_src:
    :param delta_dash:
    :return:
    """
    if delta_dash == 1.0:
        return img_src.copy()

    N,M = img_src.shape[:2]
    Ndash,Mdash = float(int(N/delta_dash)), float(int(M/delta_dash))
    img = np.zeros((int(Ndash), int(Mdash), 1), np.uint8)

    for m_dash in range(int(Mdash)):
        for n_dash in range(int(Ndash)):
            x, y = delta_dash*m_dash, delta_dash*n_dash
            x_floor, y_floor = float(int(x)), float(int(y))
            img[n_dash, m_dash] =   (x - x_floor)    *(y - y_floor)    *f_u_bar(x_floor + 1, y_floor + 1, M, N, img_src) + \
                                    (1 + x_floor - x)*(y - y_floor)    *f_u_bar(x_floor,     y_floor + 1, M, N, img_src) + \
                                    (x - x_floor)    *(1 + y_floor - y)*f_u_bar(x_floor + 1, y_floor,     M, N, img_src) + \
                                    (1 + x_floor - x)*(1 + y_floor - y)*f_u_bar(x_floor,     y_floor,     M, N, img_src)

    return img

def gen_scale_space(img, sigma, O, K):
    """

    :param img:
    :param sigma:
    :param O: number of octave
    :param K: number of images at each scale space
    :return:
    """
    scale_space = []
    img0 = img.copy()

    for o in range(1, O+1):
        img0 = bilinear_interpolation(img0, float(o))
        vec = [img0]
        for k in range(1, K+1):
            print(str(k))
        scale_space.append(vec)
    return scale_space


def main():
    path_src = "./data/lena_std.tif"

    # load image
    img_src = cv2.imread(path_src, 1)

    # convert from RGB to grayscale
    img_gray = cv2.cvtColor(img_src, cv2.COLOR_RGB2GRAY)

    # bilinear interpolation
    #delta_dash = 0.8
    #img_org = bilinear_interpolation(img_gry, delta_dash)

    #cv2.imwrite(path_src + "_org.tif", img_org)

    # scale space
    #sigma_in = 0.5
    #K = 3
    #O = 5
    #scale_space = gen_scale_space(img_org, sigma_in, O, K)

    # cv2.imshow("Image", img_gry)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    scale_space = ScaleSpace()
    scale_space.generate(img_gray)

if __name__ == "__main__":
    main()
