import numpy as np
import cv2


class Fisheye(object):
    def __init__(self, img):
        self.img = img

    def edge_detection(self):
        thresh = 10
        _, thresh = cv2.threshold(self.img, thresh, 255, cv2.THRESH_BINARY)
        height, width, _ = self.img.shape
        edge_points = [[], []]

        # top-left
        for i in range(height // 2):
            for j in range(width // 2, -1, -1):
                if thresh[i, j] != 0:
                    continue
                else:
                    edge_points[1].append(i)
                    edge_points[0].append(j)
                    break

        # bottom-left
        for i in range(height // 2, height):
            for j in range(width // 2, -1, -1):
                if thresh[i, j] != 0:
                    continue
                else:
                    edge_points[1].append(i)
                    edge_points[0].append(j)
                    break

        # bottom-right
        for i in range(height // 2, height):
            for j in range(width // 2, width):
                if thresh[i, j] != 0:
                    continue
                else:
                    edge_points[1].append(i)
                    edge_points[0].append(j)
                    break

        return edge_points

    def find_circle(self, edge_points):
        size = len(edge_points[0])
        At = edge_points.copy()
        At.append([1 for _ in range(size)])
        At = np.array(At)
        b = [x * x + y * y for x, y in zip(edge_points[0], edge_points[1])]
        b = np.array(b)
        A = np.transpose(At)
        Atb = np.matmul(At, b)
        AtA_inv = np.linalg.inv(np.matmul(At, A))
        ans = np.matmul(AtA_inv, Atb)
        return ans

    def coordinate_normalize(self, x_0, y_0, radius):
        pass

    def image_projection(self):
        pass

    def image_remapping(self):
        pass


F = Fisheye()
