import matplotlib.pyplot as plt

x_ori = [1, 2, 3, 5, 6, 10, 15, 18]
y1 = [1, 2, 3, 4, 5, 6, 7, 8]
y2 = [0.75, 1.5, 2.25, 3.5, 4.0, 5.5, 6.75, 7.5]

for i in range(8):
    y1[i] = float(y1[i])/8
    y2[i] = float(y2[i])/8

#plt.scatter(x_ori, y1, c = 'b')
plt.scatter(x_ori, y2, c = 'r')

for i in range(6):
    x = [4*i, 4*i]
    y = [0, 1]
    plt.plot(x, y, 'g')

class point:
    def __init__(self, X, Y1, Y2):
        self.x = X
        self.y1 = Y1
        self.y2 = Y2

point_list = []
for i in range(8):
    p = point(x_ori[i], y1[i], y2[i])
    point_list.append(p)
point_list = sorted(point_list, cmp=lambda a, b: cmp(a.x, b.x))

x_ori = []
y1 = []
y2 = []
for i in range(8):
    x_ori.append(point_list[i].x)
    y1.append(point_list[i].y1)
    y2.append(point_list[i].y2)

#plt.plot(x_ori, y1, 'b')
plt.plot(x_ori, y2, 'r')

plt.show()