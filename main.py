import numpy as np

edge_points = [[0, 1, 5, 3, 6], [2, 5, 7, 3, 8]]
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
print(ans)
