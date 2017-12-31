import numpy as np
import math
from numpy.linalg import cholesky
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random
import threading

class server:
    data_set = []
    def __init__(self, X_l, X_u, Y_l, Y_u, Z_l, Z_u):
        self.x_l = X_l
        self.y_l = Y_l
        self.z_l = Z_l
        self.x_u = X_u
        self.y_u = Y_u
        self.z_u = Z_u
    def publish(self, neighbor, data):
        neighbor.data_set.append(data)

class bucket:
    def __init__(self, S, T):
        self.start = S
        self.terminal = T
        self.number = 0
        self.accumulation = 0
        self.slope = 0
        self.inner_data = []
    def map(self, actual_data, last_accumulation, bucket_num, interval):
        res = last_accumulation + self.slope * (actual_data - bucket_num * interval)
        return res

class object:
    dimension_lower_bound = []
    dimension_upper_bound = []
    def __init__(self, dim, ctr, lb, ub):
        self.dimension = dim
        self.centroid = ctr
        self.lower_bound = lb
        self.upper_bound = ub

class data_point:
    def __init__(self, X, Y, Z):
        self.x = X
        self.y = Y
        self.z = Z
        self.x_mapping = 0
        self.y_mapping = 0
        self.z_mapping = 0
    affiliating_bucket = bucket(0, 0)

def two_dimension_projection(x, y, fig):
    f1 = plt.figure(fig)
    plt.subplot(211)
    plt.scatter(x, y)
    plt.show()

def one_dimension_projection(x, y, fig):
    f1 = plt.figure(fig)
    plt.subplot(211)
    #print x
    #print y
    plt.scatter(x, y)
    plt.show()

def draw_origin(x, y, z, total, fig):
    uniformity = evaluate_gap_ratio(x, y, z, total)
    discrepancy = evaluate_discrepancy(x, y, z, total)
    if fig == 1:
        print "Directly mapping gap ratio (Rp/rp): " + str(uniformity)
        print "Directly mapping discrepancy: " + str(discrepancy)
    elif fig == 2:
        print "Sorting-based gap ratio (Rp/rp): " + str(uniformity)
        print "Sorting-based discrepancy: " + str(discrepancy)
    elif fig == 3:
        print "Casting-based gap ratio (Rp/rp): " + str(uniformity)
        print "Casting-based discrepancy: " + str(discrepancy)
    elif fig == 4:
        print "Casting-based modified gap ratio (Rp/rp): " + str(uniformity)
        print "Casting-based modified discrepancy: " + str(discrepancy)
    f1 = plt.figure(fig)
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x, y, z, c='b')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()
    two_dimension_projection(x, y, 11)

def draw_sort_mapping(point_set, total):
    tmp_list = sorted(point_set, cmp=lambda a, b: cmp(a.x, b.x))
    for i in range(total):
        tmp_list[i].x_mapping = i + 1
    tmp_list = sorted(point_set, cmp=lambda a, b: cmp(a.y, b.y))
    for i in range(total):
        tmp_list[i].y_mapping = i + 1
    tmp_list = sorted(point_set, cmp=lambda a, b: cmp(a.z, b.z))
    for i in range(total):
        tmp_list[i].z_mapping = i + 1
    x = []
    x_ori = []
    y = []
    z = []
    for i in range(total):
        x.append(point_set[i].x_mapping)
        x_ori.append(point_set[i].x)
        y.append(point_set[i].y_mapping)
        z.append(point_set[i].z_mapping)
    draw_origin(x, y, z, total, 2)
    one_dimension_projection(x_ori, x, 12)


def init_bucket(partition, interval):
    bucket_set = []
    for i in range(partition):
        bucket_set.append(bucket(0, 0))
    for i in range(partition):
        bucket_set[i].start = i * interval
        bucket_set[i].terminal = (i + 1) * interval
    return bucket_set

def cast_x(point, bucket_set, interval):
    bucket_set[int(point.x / interval)].inner_data.append(point)
    bucket_set[int(point.x / interval)].number += 1

def cast_y(point, bucket_set, interval):
    bucket_set[int(point.y / interval)].inner_data.append(point)
    bucket_set[int(point.y / interval)].number += 1

def cast_z(point, bucket_set, interval):
    bucket_set[int(point.z / interval)].inner_data.append(point)
    bucket_set[int(point.z / interval)].number += 1

def process_bucket(bucket_set, partition, interval, dimension):
    for i in range(partition):
        if i == 0:
            bucket_set[i].accumulation = bucket_set[i].number
        else:
            bucket_set[i].accumulation = bucket_set[i - 1].accumulation + bucket_set[i].number
        bucket_set[i].slope = float(bucket_set[i].number)/interval
        for point in bucket_set[i].inner_data:
            if dimension == 1:
                point.x_mapping = bucket_set[i].map(point.x, bucket_set[i - 1].accumulation, i, interval)
            elif dimension == 2:
                point.y_mapping = bucket_set[i].map(point.y, bucket_set[i - 1].accumulation, i, interval)
            elif dimension == 3:
                point.z_mapping = bucket_set[i].map(point.z, bucket_set[i - 1].accumulation, i, interval)


def dec2bin(x, total):
    INT = bin(int(x))
    INT = str(INT)[2:]
    LEN = len(str(bin(total))[2:])
    while LEN%3 != 0:
        LEN += 1
    while len(INT) < LEN:
        INT = '0' + INT
    x -= int(x)
    bins = ''
    while x:
        x *= 2
        bins = bins + str(1 if x >= 1. else 0)
        x -= int(x)
        if len(bins) >= LEN:
            break
    while len(bins) < LEN:
        bins += '0'
    return INT + bins, LEN

def bin2dec(x, iteration):
    result = 0.0
    for i in range(iteration):
        result += int(x[iteration - i - 1])*(2**i)
    for i in range(iteration):
        result += int(x[iteration + i])*((1.0/2)**(i + 1))
    return result

def homogenize (x, total):
    x_homo = []
    y_homo = []
    z_homo = []
    class bijection:
        def __init__(self, number, total):
            NUM, LEN = dec2bin(number, total)
            self.iteration = LEN/3
            c = ''
            b = ''
            a = ''
            for i in range(2 * self.iteration):
                a = a + NUM[3 * i]
                b = b + NUM[3 * i + 1]
                c = c + NUM[3 * i + 2]
                #print "( " + str(self.a) + ", " + str(self.b) + ", " + str(self.c) + ")"
            self.a = bin2dec(a, self.iteration)
            self.b = bin2dec(b, self.iteration)
            self.c = bin2dec(c, self.iteration)
    for i in range(total):
        triplet = bijection(x[i], total)
        x_homo.append(triplet.a*(2**(2*triplet.iteration )))
        y_homo.append(triplet.b*(2**(2*triplet.iteration )))
        z_homo.append(triplet.c*(2**(2*triplet.iteration )))
    return x_homo, y_homo, z_homo

def calculate_uniformity(bucket_set):
    result = 1
    for i in bucket_set:
        result = result * i.slope
    return result

def draw_casting_buckets(point_set, partition, scope, total):
    interval = scope/partition
    bucket_set1 = init_bucket(partition, interval)
    bucket_set2 = init_bucket(partition, interval)
    bucket_set3 = init_bucket(partition, interval)
    for i in range(total):
        current_point = point_set[i]
        cast_x(current_point, bucket_set1, interval)
        cast_y(current_point, bucket_set2, interval)
        cast_z(current_point, bucket_set3, interval)
    process_bucket(bucket_set1, partition, interval, 1)
    process_bucket(bucket_set2, partition, interval, 2)
    process_bucket(bucket_set3, partition, interval, 3)
    x = []
    x_ori = []
    y = []
    z = []
    for i in range(total):
        x.append(point_set[i].x_mapping)
        x_ori.append(point_set[i].x)
        y.append(point_set[i].y_mapping)
        z.append(point_set[i].z_mapping)
    draw_origin(x, y, z, total, 3)
    #one_dimension_projection(x_ori, x, 32)

def draw_casting_and_modifying(point_set, partition, scope, total):
    interval = scope/partition
    bucket_set1 = init_bucket(partition, interval)
    bucket_set2 = init_bucket(partition, interval)
    bucket_set3 = init_bucket(partition, interval)
    for i in range(total):
        current_point = point_set[i]
        cast_x(current_point, bucket_set1, interval)
        cast_y(current_point, bucket_set2, interval)
        cast_z(current_point, bucket_set3, interval)
    process_bucket(bucket_set1, partition, interval, 1)
    process_bucket(bucket_set2, partition, interval, 2)
    process_bucket(bucket_set3, partition, interval, 3)
    x = []
    x_ori = []
    y = []
    z = []
    uniformity1 = calculate_uniformity(bucket_set1)  # Calculate the product of the slopes of all buckets in the set
    uniformity2 = calculate_uniformity(bucket_set2)
    uniformity3 = calculate_uniformity(bucket_set3)
    winner = max(uniformity1, uniformity2, uniformity3)# We use the dimension with the best uniformity
    for i in range(total):
        x.append(point_set[i].x_mapping)
        x_ori.append(point_set[i].x)
        y.append(point_set[i].y_mapping)
        z.append(point_set[i].z_mapping)
    if winner == uniformity1:
        x, y, z = homogenize(x, total)
    elif winner == uniformity2:
        x, y, z = homogenize(y, total)
    elif winner == uniformity3:
        x, y, z = homogenize(z, total)
    draw_origin(x, y, z, total, 4)
    #one_dimension_projection(x_ori, x, 32)

def generate_points(List, scope1, scope2):
    for i in range(scope1):
        for j in range(scope2):
            List.append(random.uniform(1, scope2 * (i)))

def generate_points4(List, scope):
    for i in range(scope):
        List.append(random.uniform(1, i))

def generate_points3(List, scope1, scope2):
    for i in range(scope1):
        for j in range(scope2):
            rdm = int(random.uniform(0, 1) * 2)
            if rdm == 1:
                List.append(random.uniform(1, scope2 * (i)))
            else:
                List.append(500 - random.uniform(1, scope2 * (i)))

def generate_points2(mu, sigma, total):
    np.random.seed(0)
    '''lst = []
        for i in range(50):
        lst = lst + list(np.random.normal(mu, sigma*(i + 1), 10))'''
    lst = list(np.random.normal(mu, sigma, total))
    MAX = 0
    for i in lst:
        if abs(i - 250) > MAX:
            MAX = abs(i - 250)
    for i in range(500):
        lst[i] = 250 + float(lst[i] - 250)*250/MAX
    return lst

def calculate_euclidean_distance(x1, y1, z1, x2, y2, z2):
    result = ((x1 - x2)**2 + (y1 - y2)**2 +(z1 - z2)**2)**(1.0/2)
    return result

def evaluate_gap_ratio(x, y, z, total):
    distance = np.zeros((total,total))
    for i in range(total):
        for j in range(total):
            distance[i][j] = calculate_euclidean_distance(x[i], y[i], z[i], x[j], y[j], z[j])
            if distance[i][j] == 0:
                distance[i][j] = 1000
    rp = np.min(distance)/2
    print "===================================="
    print "rp: " + str(rp)
    corner_x = [0, 0, 0, 0, total, total, total, total]
    corner_y = [0, 0, total, total, 0, 0, total, total]
    corner_z = [0, total, 0, total, 0, total, 0, total]
    distance = np.zeros((500, 8))
    for i in range(500):
        for j in range(8):
            distance[i][j] = calculate_euclidean_distance(x[i], y[i], z[i], corner_x[j], corner_y[j], corner_z[j])
    Rp = np.max(distance)
    print "Rp: " + str(Rp)
    return Rp/rp

def evaluate_discrepancy(x, y, z, total):
    point_set = []
    for i in range(total):
        point = data_point(x[i], y[i], z[i])
        point_set.append(point)
    tmp_list = sorted(point_set, cmp=lambda a, b: cmp(a.x, b.x))
    x = sorted(x)
    y = sorted(y)
    z = sorted(z)
    MAX = 0
    for j in y:
        for k in x:
            count = 0
            for point in tmp_list:
                if point.x > k:
                    discrepancy = abs(float(j * k) / (500 ** 2) - float(count) / total)
                    if discrepancy > MAX:
                        MAX = discrepancy
                    break
                if point.y <= j:
                    count += 1
    return MAX

point_set = []
x = []
y = []
z = []
#Generate a data set in overlapping uniform distributions
generate_points(x, 50, 10)
generate_points(y, 50, 10)
generate_points(z, 50, 10)

#Another data set in opverlapping normal distribution and multiple uniform distributions
'''x = generate_points2(250, 100, 500)
y = []
generate_points(y, 50, 10)
z = []
generate_points4(z, 500)'''


for i in range(500):
    point = data_point(x[i], y[i], z[i])
    point_set.append(point)

draw_origin(x, y, z, 500, 1)
draw_sort_mapping(point_set, 500)
draw_casting_buckets(point_set, 10, 500, 500)
draw_casting_and_modifying(point_set, 10, 500, 500)




