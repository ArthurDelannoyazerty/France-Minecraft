import numpy as np



a = np.array([[0, 0, 0],
              [0, 0, 0],
              [0, 0, 1],])
print('a')
print(a)
print(a.shape)

b0 = np.unique(a, axis=0)
b1 = np.unique(a, axis=1)
bnone = np.unique(a)


print('b0')
print(b0)
print(b0.shape)
print('b1')
print(b1)
print(b1.shape)
print('bnone')
print(bnone)
print(bnone.shape)