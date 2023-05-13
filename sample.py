import cv2
import numpy as np
import math


# import matplotlib.pyplot as plt


class Fisheye(object):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def edge_detection(self, img, thresh):
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

        return np.array(edge_points)

    def find_circle(self, edge):
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

    def lat_lon_to_vector(self, height, width):
        x = np.arange(0, width)
        y = np.arange(0, height)
        [x, y] = np.meshgrid(x, y)
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)
        lat_lon = np.hstack((x, y))

        return lat_lon

    def lat_lon_to_sphere(self, lat_lon, height, width):
        points_sphere = []
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
            points_sphere.append([x, y, z])
        return points_sphere

    def sphere_to_image(self, points_sphere, flag):
        image_points = []
        for point in points_sphere:
            x = point[0]
            y = point[1]
            z = point[2]
            theta = math.atan(math.sqrt(x * x + y * y) / z)

            if flag == "equidistant":
                rho = 2 * theta / math.pi
            elif flag == "orthogonal":
                rho = math.sin(theta)
            elif flag == "equiangular":
                rho = math.sqrt(2) * math.sin(theta / 2)
            elif flag == "stereographic":
                rho = math.tan(theta / 2)

            alpha = math.atan2(y, x)
            img_x = rho * math.cos(alpha)
            img_y = rho * math.sin(alpha)
            image_points.append([img_x, img_y])
        return image_points

    def image_to_pixel(self, image_points, x_0, y_0, radius):
        uv = []
        for point in image_points:
            x = point[0]
            y = point[1]
            u = x_0 + radius * x
            v = y_0 + radius * y
            uv.append([u, v])
        return uv

    def project_fisheye_image(self, img, flag):
        points = self.edge_detection(img, 10)
        x0, y0, radius = self.find_circle(points)
        lat_lon = self.lat_lon_to_vector(self.height, self.width)
        points_sphere = self.lat_lon_to_sphere(lat_lon, self.height, self.width)
        image_points = self.sphere_to_image(points_sphere, flag)
        uv = self.image_to_pixel(image_points, x0, y0, radius)
        res = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for img_points, lat_lon in zip(uv, lat_lon):
            col = int(img_points[0])
            row = int(img_points[1])
            c = int(lat_lon[0])
            r = int(lat_lon[1])
            if 0 <= row < 1200 and 0 <= col < 1200:
                res[r, c, :] = img[row, col, :]
                # res[r, c, 0] = img[row, col, 0]
                # res[r, c, 1] = img[row, col, 1]
                # res[r, c, 2] = img[row, col, 2]

        return res


if __name__ == "__main__":
    img = cv2.imread("./figs/fig3.jpg")
    flag = "orthogonal"
    F = Fisheye(1920, 1080)
    res = F.project_fisheye_image(img, flag)
    cv2.imwrite(flag + ".jpg", res)
    # cv2.namedWindow("img", cv2.WINDOW_KEEPRATIO)
    # cv2.imshow("img", res)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
