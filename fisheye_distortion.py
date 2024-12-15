import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
# import numba
# from numba import jit


def get_all_coordinate(img):
    height, width, _ = img.shape
    x = np.arange(0, width)
    y = np.arange(0, height)
    [x, y] = np.meshgrid(x, y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    coordinate = np.hstack((x, y))

    return coordinate


def distort_coordinate(radius_distort):
    k1_lower = 1e-4
    k1_upper = 1e-6
    k2_lower = 1e-9
    k2_upper = 1e-11

    k1 = np.random.uniform(k1_lower, k1_upper)
    k2 = np.random.uniform(k2_lower, k2_upper)

    p = np.poly1d([k2, 0, k1, 0, 1, -radius_distort])
    radius = p.roots[-1]

    return radius, k1, k2


# @jit(nopython=True)
def distort_image(height, width, img, k1, k2):
    dst = np.zeros((height, width, 3), np.uint8)

    x0 = (width - 1) / 2
    y0 = (height - 1) / 2

    h, w = img.shape[:2]
    x1 = (w - 1) / 2
    y1 = (h - 1) / 2

    for i in range(width):
        for j in range(height):
            xd = i - x0
            yd = j - y0

            rd = np.sqrt(xd * xd + yd * yd)
            r = np.real(np.roots([k2, 0, k1, 0, 1, -rd])[-1])

            x = r * xd / rd
            y = r * yd / rd

            u = int(x + x1)
            v = int(y + y1)

            if 0 <= u < w and 0 <= v < h:
                dst[j, i, :] = img[v, u, :]

    # img_height, img_width, _ = img.shape
    # cx = img_height // 2
    # cy = img_width // 2

    # for c in coordinate:
    #     i, j = c[0], c[1]
    #     radius_distort = np.sqrt((i - cx) * (i - cx) + (j - cy) * (j - cy))
    #     radius, k1, k2 = distort_coordinate(radius_distort)
    #     factor = 1 + k1 * radius**2 + k2 * radius**4
    #     row = int(j / factor)
    #     col = int(i / factor)
    #     if 0 <= row < img_height and 0 <= col < img_width:
    #         dst[j, i, :] = img[row, col, :]

    return dst


if __name__ == "__main__":
    k1_lower = 1e-4
    k1_upper = 1e-6
    k2_lower = 1e-9
    k2_upper = 1e-11

    k1 = np.random.uniform(k1_lower, k1_upper)
    k2 = np.random.uniform(k2_lower, k2_upper)

    filename = os.getcwd() + "\\figs\\0166.jpg"
    img = cv2.imread(filename)

    # coordinate = get_all_coordinate(img)

    dst = distort_image(480, 640, img, k1, k2)

    cv2.imshow("dst", dst)
    cv2.imwrite("./0166.jpg", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
