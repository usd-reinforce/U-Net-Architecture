import cv2
import numpy as np
from scipy.signal import wiener
from skimage.morphology import binary_dilation, square, dilation
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm

"""
create def(s) that contains of Bothat and Laplacian preprocessing

Bothat:
- convert to grayscale *
- Average image filtering
- Laplacian image filtering *
- Subtract filtered images (average image and laplacian image filtering) 
- Bottom hat transform se0
- Bottom hat transform se45
- Bottom hat transform se90
- Add images (se0 + se45 + se90)
- Image adjustment
- Global image thresholding to obtain binary mask
- Image dilation  *
- Red channel hair pixel replacement using interpolation *
- Green channel hair pixel replacement using interpolation *
- Blue channel hair pixel replacement using interpolation *
- Combine 3 of them to produce final image *

Laplacian:
- convert to grayscale *
- Laplacian image filtering *
- Subtract filtered images (grayscale and laplacian image)
- Noise reduction filtering
- Obtain binary mask using log edge detection
- Morphological operations
- Morphological image closing se0
- Morphological image closing se45
- Morphological image closing se90
- Image dilation *
- Red channel hair pixel replacement using interpolation *
- Green channel hair pixel replacement using interpolation *
- Blue channel hair pixel replacement using interpolation *
- Combine 3 of them to produce final image *
"""

def Laplacian_hr(image_path):
    img = cv2.imread(image_path)

class Laplacian:
    def __init__(self):
        self.img = None
        self.gray = None
        self.mask = None
        self.laplacian = None
        self.subtracted = None
        self.reduced = None
        self.edge = None
        self.morp = None
        self.result = None

    def __to_grayscale__(self):
        self.gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        return self.gray

    def __laplacian_filtering__(self):
        """
        This def process grayscale image by sharpening filter based on laplace operation
        """

        self.laplacian = cv2.Laplacian(self.gray, cv2.CV_64F)
        self.laplacian = np.uint8(np.absolute(self.laplacian))
        return self.laplacian

    def __subtract__(self):
        """
        subtract two processed image from grayscale and laplacian filtering method
        :return: subtracted image
        """
        self.subtracted = cv2.subtract(self.gray, self.laplacian)
        return self.subtracted

    def __noise_reduction__(self):
        """
        reduce subtracted image with 3x3 wiener filter
        """
        self.reduced = wiener(self.subtracted, (3, 3))
        self.reduced = np.uint8(np.clip(self.subtracted, 0, 255))
        return self.reduced

    def __log_binary_mask__(self):
        """
        cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge detection using laplacian of gaussian (log)
        this operation will process image with two operations,
        the one is gaussian that decrease artifact,
        and the other one is laplace that using laplacian mask to
        minimize error in edge detection
        """

        blur = cv2.GaussianBlur(self.reduced, (3, 3), 0)
        self.edge = cv2.Laplacian(blur, cv2.CV_64F)
        self.edge = np.uint8(np.absolute(self.edge))
        return self.edge

    def __morphological_operation__(self):
        # bridge
        bridged = binary_dilation(self.edge, square(2)) & self.edge

        # diagonal
        diagonal = binary_dilation(bridged, square(3))

        # clean
        clean = None
        

    def __image_closing__(self):
        pass

    def __image_dilation__(self):
        pass

    def __interpolation__(self):
        pass

    def process(self, image: np.ndarray):
        self.__to_grayscale__()
        self.__laplacian_filtering__()
        self.__subtract__()
        self.__noise_reduction__()
        self.__log_binary_mask__()
        self.__morphological_operation__()
        self.__image_closing__()
        self.__image_dilation__()
        self.__interpolation__()

        return self.result
