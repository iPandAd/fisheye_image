import numpy as np

op1 = np.array([i for i in range(9)]).reshape(3, 3)
op2 = np.array([[1, 2, 3]])
op3 = np.array([1, 2, 3])
op4 = np.array([3, 2, 1])

print(op2)
print(op2.T)

initialize_2d_list = [[i + j for i in range(5)] for j in range(9)]
print(initialize_2d_list)
