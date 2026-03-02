import cv2
import os
import numpy as np
from scipy.signal import wiener
import matplotlib.pyplot as plt

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

def laplacian_hr(input_data, debug=False):
    # read an image

    if isinstance(input_data, str):
        img = cv2.imread(input_data)
    else:
        img = input_data

    # convert read image into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # apply read image into laplacian operation
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_64f = cv2.convertScaleAbs(laplacian)

    # subtract gray and laplacian images
    subtracted = cv2.subtract(gray, laplacian_64f)
    
    # apply wiener filter with 3x3 kernel to reduce subtracted image
    reduced = wiener(subtracted, (3, 3))
    reduced = np.nan_to_num(reduced)
    reduced = np.clip(reduced, 0, 255)
    reduced = np.uint8(reduced)

    # log binary mask
    blur = cv2.GaussianBlur(reduced, (3, 3), 0)
    log_edges = cv2.Laplacian(blur, cv2.CV_64F)
    log_edges_8u = cv2.convertScaleAbs(log_edges)
    _, binary_mask = cv2.threshold(log_edges_8u, 15, 255, cv2.THRESH_BINARY)

    # morphological operation (clean)
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_clean = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel_clean)

    # morphological operation (bridge) & (diag)
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_bridge = cv2.morphologyEx(img_clean, cv2.MORPH_CLOSE, kernel_bridge)

    # se0 = Horizontal, se45 = diagonal, se90 = vertical
    se0 = np.array([[0,0,0],[1,1,1],[0,0,0]], dtype=np.uint8)
    se45 = np.array([[0,0,1],[0,1,0],[1,0,0]], dtype=np.uint8)
    se90 = np.array([[0,1,0],[0,1,0],[0,1,0]], dtype=np.uint8)

    # apply se0
    img_se0 = cv2.morphologyEx(img_bridge, cv2.MORPH_CLOSE, se0)
    img_se45 = cv2.morphologyEx(img_se0, cv2.MORPH_CLOSE, se45)
    img_se90 = cv2.morphologyEx(img_se45, cv2.MORPH_CLOSE, se90)

    # dilation kernel
    kernel_dilation = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_dilate = cv2.dilate(img_se90, kernel_dilation, iterations=1)

    # interploation

    # inpainting / restoration
    final_img = cv2.inpaint(img, img_dilate, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    if debug:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(3, 4, figsize=(20, 10))

        ax[0, 0].imshow(img_rgb)
        ax[0, 0].set_title("original image")
        ax[0, 1].imshow(gray, cmap='gray')
        ax[0, 1].set_title("grayscale image")
        ax[0, 2].imshow(laplacian_64f, cmap='gray')
        ax[0, 2].set_title("laplacian image filter")
        ax[0, 3].imshow(reduced, cmap='gray')
        ax[0, 3].set_title("weiner")
        ax[1, 0].imshow(binary_mask, cmap='gray')
        ax[1, 0].set_title("LoG edge detection")
        ax[1, 1].imshow(img_clean, cmap='gray')
        ax[1, 1].set_title("Morphological operation clean")
        ax[1, 2].imshow(img_bridge, cmap='gray')
        ax[1, 2].set_title("Morphological operation bridge")
        ax[1, 3].imshow(img_se0, cmap='gray')
        ax[1, 3].set_title("se0")
        ax[2, 0].imshow(img_se45, cmap='gray')
        ax[2, 0].set_title("se45")
        ax[2, 1].imshow(img_se90, cmap='gray')
        ax[2, 1].set_title("se90")
        ax[2, 2].imshow(img_dilate, cmap='gray')
        ax[2, 2].set_title("dilated")
        ax[2, 3].imshow(final_rgb)
        ax[2, 3].set_title("result")

        for a in ax.flat:
            a.axis('off')

        plt.tight_layout()
        plt.show()

    return final_img, img_dilate

def bothat_hr(input_data, debug=False):
    if isinstance(input_data, str):
        img = cv2.imread(input_data)
    else:
        img = input_data

    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # average image filtering
    kernel_size = [3, 3]
    blurred_img = cv2.blur(gray, kernel_size)

    # laplacian image filtering
    laplacian = cv2.Laplacian(blurred_img, cv2.CV_64F)
    laplacian_64f = cv2.convertScaleAbs(laplacian)

    # subtract average image filtering with laplacian image filtering
    subtracted = cv2.subtract(blurred_img, laplacian_64f)

    # transform se0, se45, se90
    se0 = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.uint8)
    se45 = np.array([[0, 0, 0, 0, 1], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.uint8)
    se90 = np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)

    img_se0 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se0)
    img_se45 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se45)
    img_se90 = cv2.morphologyEx(subtracted, cv2.MORPH_BLACKHAT, se90)

    # add images
    add_1 = cv2.add(se0, se45)
    add_2 = cv2.add(add_1, se90)

    # adjustment image
    adj_img = cv2.convertScaleAbs(add_2, alpha=1.5)

    # image adjustment

    # global image thresholding

    # image dilation

    # Red, Green, Blue channel hair pixel replacement using interpolation
