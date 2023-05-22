import cv2
import numpy as np
import math

import numba
from numba import jit
import time


# import matplotlib.pyplot as plt


# @jit(nopython=True)
def edge_detection(img, thresh):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    height, width = th.shape
    edge_points = []

    # top
    for j in range(width):
        for i in range(height // 2):
            if th[i, j] > 0:
                edge_points.append([j, i])
                break

    # bottom
    for j in range(width):
        for i in range(height - 1, height // 2, -1):
            if th[i, j] > 0:
                edge_points.append([j, i])
                break

    # for point in edge_points:
    #     cv2.circle(img, point, 1, (0, 255, 0))
    # plt.imshow(img)
    # plt.show()

    return np.array(edge_points)


# @jit(nopython=True)
def find_circle(edge):
    ones = np.ones((edge.shape[0], 1))
    a = np.hstack((edge, ones))
    a_t = np.transpose(a)
    ata = np.dot(a_t, a)
    square = np.square(edge[:, 0]) + np.square(edge[:, 1])
    atb = np.dot(a_t, square)
    inv = np.linalg.inv(ata)
    sln = np.dot(inv, atb)
    x_0 = sln[0] // 2
    y_0 = sln[1] // 2
    radius = np.sqrt(sln[2] + (sln[0] / 2) ** 2 + (sln[1] / 2) ** 2)

    return x_0, y_0, radius


# @jit(nopython=True)
def lat_lon_to_vector(height, width):
    x = np.arange(0, width)
    y = np.arange(0, height)
    [x, y] = np.meshgrid(x, y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    lat_lon = np.hstack((x, y))

    return lat_lon


@jit(nopython=True)
def lat_lon_to_sphere(lat_lon, height, width):
    points_sphere = np.zeros((lat_lon.shape[0], 3))

    # points_sphere = []
    delta_x = math.pi / width
    delta_y = math.pi / height
    for i in range(lat_lon.shape[0]):
        lam = lat_lon[i, 0] * delta_x
        phi = lat_lon[i, 1] * delta_y
        # x = math.sin(phi + math.pi) * math.cos(lam + math.pi)
        # y = math.cos(phi + math.pi)
        # z = math.sin(phi + math.pi) * math.sin(lam + math.pi)

        x = math.sin(math.pi - phi) * math.cos(math.pi - lam)
        y = math.cos(math.pi - phi)
        z = math.sin(math.pi - phi) * math.sin(math.pi - lam)
        points_sphere[i, :] = np.array([x, y, z])
    return points_sphere


@jit(nopython=True)
def sphere_to_image(points_sphere, flag):
    # image_points = []

    image_points = np.zeros((points_sphere.shape[0], 2))
    for i in range(points_sphere.shape[0]):
        x, y, z = points_sphere[i, :]
        # x = points_sphere[i, 0]
        # y = points_sphere[i, 1]
        # z = points_sphere[i, 2]
        theta = np.arccos(z)
        # theta = np.atan(np.sqrt(x * x + y * y) / z)

        if flag == "equidistant":
            rho = 2 * theta / np.pi
        elif flag == "orthogonal":
            rho = np.sin(theta)
        elif flag == "equiangular":
            rho = np.sqrt(2) * np.sin(theta / 2)
        elif flag == "stereographic":
            rho = np.tan(theta / 2)

        alpha = np.arctan2(y, x)
        img_x = rho * np.cos(alpha)
        img_y = rho * np.sin(alpha)
        # image_points.append([img_x, img_y])
        image_points[i, :] = np.array([img_x, img_y])
    return image_points


@jit(nopython=True)
def image_to_pixel(image_points, x_0, y_0, radius):
    # uv = []
    uv = np.zeros((points_sphere.shape[0], 2))
    for i in range(image_points.shape[0]):
        x = image_points[i, 0]
        y = image_points[i, 1]
        u = x_0 + radius * x
        v = y_0 + radius * y
        uv[i, :] = np.array([u, v])
    return uv


@jit(nopython=True)
def fill_image(uv, lat_lon, img):
    res = np.zeros((1080, 1920, 3), dtype=np.uint8)

    for i in range(uv.shape[0]):
        col = int(uv[i][0])
        row = int(uv[i][1])
        c = int(lat_lon[i][0])
        r = int(lat_lon[i][1])
        if 0 <= row < 1200 and 0 <= col < 1200:
            res[r, c, :] = img[row, col, :]

    return res


if __name__ == "__main__":
    img = cv2.imread("./figs/fig3.jpg")

    t_0 = time.perf_counter()
    points = edge_detection(img, 10)

    t_1 = time.perf_counter()
    print(f"Time1: {t_1 - t_0}")

    x0, y0, radius = find_circle(points)

    t_2 = time.perf_counter()
    print(f"Time2: {t_2 - t_1}")

    lat_lon = lat_lon_to_vector(1080, 1920)

    t_3 = time.perf_counter()
    print(f"Time3: {t_3 - t_2}")

    points_sphere = lat_lon_to_sphere(lat_lon, 1080, 1920)

    t_4 = time.perf_counter()
    print(f"Time4: {t_4 - t_3}")

    flag = "orthogonal"
    image_points = sphere_to_image(points_sphere, flag)

    t_5 = time.perf_counter()
    print(f"Time5: {t_5 - t_4}")
    uv = image_to_pixel(image_points, x0, y0, radius)

    t_6 = time.perf_counter()
    print(f"Time6: {t_6 - t_5}")

    res = fill_image(uv, lat_lon, img)

    # res[r, c, 1] = img[row, col, 1]
    # res[r, c, 2] = img[row, col, 2]

    # for img_points, lat_lon in zip(uv, lat_lon):
    #     col = int(img_points[0])
    #     row = int(img_points[1])
    #     c = int(lat_lon[0])
    #     r = int(lat_lon[1])
    #     if 0 <= row < 1200 and 0 <= col < 1200:
    #         res[r, c, 0] = img[row, col, 0]
    #         res[r, c, 1] = img[row, col, 1]
    #         res[r, c, 2] = img[row, col, 2]

    t_7 = time.perf_counter()

    print(f"Time7: {t_7 - t_6}")
    cv2.imwrite(flag + ".jpg", res)
    # cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("img", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
