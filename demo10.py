import numpy as np

a = np.zeros((10, 2))
print(a.shape)
b = a.T
print(b.shape)
c = b.view()
print(c.shape)
d = np.reshape(b, (5, 4))
print(d.shape)
e = np.reshape(b, (20,))
print(e.shape)
print(e)
f = np.reshape(b, (20, -1))
print(f.shape)
g = np.reshape(b, (-1, 20))
print(g.shape)
print(d)