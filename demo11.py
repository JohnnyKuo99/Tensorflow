import numpy as np

a = np.array([[1, 2], [3, 4]])
b = a.view()
c = a
print(a)
print(b)
print(c)
print('change b shape')
b.shape = (4, -1)
print(a)
print(b)
print(c)
print('change c, = change a')
a.shape = (1, 4)
print(a)
print(b)
print(c)
a[0][0] = 100
print(a)
print(b)
print(c)