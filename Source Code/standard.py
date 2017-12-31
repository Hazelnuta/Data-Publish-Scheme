from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from random import shuffle
import random
import numpy as np
from numpy.linalg import cholesky

x = []
y = []
z = []

for i in range(100):
    x.append(i+1)
    y.append(i+1)
    z.append(i+1)

'''
for i in range(50):
    list = np.random.normal(10*(i+1), 2, 10)
    for item in list:
        z.append(item)

for i in range(500):
    x.append(random.uniform(0,500))
    y.append(random.uniform(0,500))'''

shuffle(x)
shuffle(y)
shuffle(z)


ax = plt.subplot(111, projection='3d')
ax.scatter(x, y, z, c='b')
ax.set_zlabel('Z')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()