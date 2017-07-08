# coding: utf-8

import cv2
import numpy as np
import pprint

class ScaleSpace(object):
    def __init__(self, K = 3, O = 8, sigma_0 = 0.8, delta_0 = 0.5):
        """
        This class is model of Gaussian Scale space
        :param K: number of scales per octave
        :param O: number of octaves
        :param sigma_0: initial value of sigma
        :param delta_0 initial ratio of subsampling image
        """
        self.K = K
        self.O = O
        self.sigma_0 = sigma_0
        self.delta_0 = delta_0
        self.image_0 = None
        self.images = {}

    def generate(self, image_in):
        """
        generate gaussian scale space
        :param image_in:
        :return:
        """
        # initialize this object
        self._init(image_in)

        # generate uin
        self.image_0 = self._gen_image_0(image_in, self.delta_0)
        cv2.imwrite("./data/lena_std_org.tif", self.image_0)

        # generate each octave
        for o in range(self.O):

            # set 1st image in o th octave
            if o == 0:
                self.images[o][0] = self._do_gaussian(self.image_0, self.sigma_0)
            else:
                self.images[o][0] = self._gen_image_0(self.images[o-1][self.K], o+1)

            cv2.imwrite("./data/g_scale_space/lena_std_" + str(o) + "_" + str(0) + ".tif", self.images[o][0])

            for k in range(1, self.K + 3):
                sigma = np.float_power(2.0, float(k)/float(self.K)) * self.sigma_0
                self.images[o][k] = self._do_gaussian(self.images[o][k-1], sigma)
                cv2.imwrite("./data/g_scale_space/lena_std_" + str(o) + "_" + str(k) + ".tif", self.images[o][k])

    def _gen_image_0(self, image, delta_0):
        """

        :param image:
        :param delta_0:
        :return:
        """
        h0, w0 = image.shape[:2]
        h, w = int(h0/delta_0), int(w0/delta_0)
        image_0 = cv2.resize(image, (h, w), interpolation=cv2.INTER_LINEAR)
        return image_0

    def _do_gaussian(self, image, sigma):
        """

        :param image:
        :param sigma:
        :return:
        """
        #return cv2.GaussianBlur(image, (length, length), sigma)

        kernel = []
        y_min = -4.0*sigma
        x_min = -4.0*sigma
        length = 2*int(4*sigma)+1
        a = 1.0/(2.0*np.pi*sigma*sigma)
        b = -1.0/(2.0*sigma*sigma)

        for row in range(length):
            rows = []
            y = float(row) + y_min
            y2 = y*y
            for col in range(length):
                x = float(col) + x_min
                x2 = x*x
                gauss = a*np.exp(b*(x2 + y2))
                rows.append(gauss)
            kernel.append(rows)

        return cv2.filter2D(image, -1, np.array(kernel))

    def _init(self, image):
        """
        initialize images
        :return:
        """
        o_max = self.O
        h0, w0 = image.shape[:2]
        h, w = int(h0 / self.delta_0), int(w0 / self.delta_0)

        for o in range(self.O):
            o_dash = o + 1
            h_dash, w_dash = int(h/o_dash), int(w/o_dash)
            pprint.pprint([h_dash, w_dash])
            if h_dash == 0 or w_dash == 0:
                o_max = o - 1
        self.O = o_max - 1

        # init O
        for o in range(self.O):
            dic = {}
            for k in range(self.K+2):
                dic[k] = None
            self.images[o] = dic