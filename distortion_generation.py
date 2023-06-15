import cv2
import numpy as np
import os


def get_distortion_size(img):
    height, width, _ = img.shape
    cx, cy = (width - 1) / 2, (height - 1) / 2
    rd = np.sqrt(cx * cx + cy * cy)
    r = np.real(np.roots([k2, 0, k1, 0, 1, -rd])[-1])
    w1 = int((width - 1) * r / rd)
    h1 = int((height - 1) * r / rd)
    return [h1, w1]


def distort_image(shape, img, k1, k2):
    dst = np.zeros((shape[0], shape[1], 3), np.uint8)
    h, w, _ = img.shape
    cx, cy = (shape[1] - 1) // 2, (shape[0] - 1) // 2
    x0 = (w - 1) / 2
    y0 = (h - 1) / 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            r = np.sqrt((i - cy) * (i - cy) + (j - cx) * (j - cx))
            factor = 1 + k1 * r * r + k2 * r * r * r * r
            xd = (j - cx) * factor
            yd = (i - cy) * factor
            u = int(xd + x0)
            v = int(yd + y0)
            if u in range(w) and v in range(h):
                dst[i, j, :] = img[v, u, :]
    return dst


if __name__ == "__main__":
    k1_lower = 1e-4
    k1_upper = 1e-6
    k2_lower = 1e-9
    k2_upper = 1e-11

    k1 = np.random.uniform(k1_lower, k1_upper)
    k2 = np.random.uniform(k2_lower, k2_upper)

    filename = os.getcwd() + "\\figs\\2399.jpg"
    img = cv2.imread(filename)
    shape = get_distortion_size(img)
    dst = distort_image(shape, img, k1, k2)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
