import numpy as np

a1 = [1, 2]
a2 = [3, 4]
a3 = [5, 6]
print(np.c_[5, 6, 7])
print(np.c_[a1, a2, a3])
print(np.r_[5, 6, 7])
print(np.r_[a1, a2, a3])
print("using stack")
print(np.hstack((np.array(a1), np.array(a2), np.array(a3))))
print(np.vstack((a1, a2, a3)))