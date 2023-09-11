import numpy as np
import cv2 as cv
import math
import numpy as np
import csv
import scipy
from os import listdir
import matplotlib.pyplot as plt

def line_to_point_distance(p, q, r):
    d = np.linalg.norm(np.cross(q-p, p-r))/np.linalg.norm(q-p)
    return d

lines = []
normalizedcenters = []
with open("old results/UWB/120.9.txt") as file:
    lines = file.readlines()

uwbData = []
        
for line in lines:
    x = line.split(",")
    uwbData.append((x[1], x[2]))

with open("results/120.9.MP4.csv") as file:
    normalizedcenters1 = csv.reader(file)
    for row in normalizedcenters1:
        print(row)
        if row != []:
            row[0] = float(row[0])
            row[1] = float(row[1])
            if(float(row[0]) < 0):
                row[0] = row[0] * -1
            if(float(row[1]) < 0):
                row[1] = row[1] * -1
            normalizedcenters.append((float(row[0]), float(row[1])))

print(normalizedcenters)


val = input("Enter height of drone: ")
heights = []
print(val)
for x in uwbData:
    closest = []
    x = (float(x[0]), float(x[1]))
    for g in normalizedcenters:
        closest.append(line_to_point_distance(np.array([0.0, 0.0]), np.array([x[0], x[1]]), np.array([float(g[0]), float(g[1])])))
    index = closest.index(min(closest))
    point = normalizedcenters[index]
    R = math.dist(point, (0.0, 0.0))
    R1 = math.dist(point, x)
    height = float(val) * float(R1) / float(R)
    heights.append(height)
print(heights)
print(np.shape(heights))
print(len(uwbData))
print(len(normalizedcenters))
hi = np.asarray(uwbData)
print(np.shape(hi))
            
fig = plt.figure()
ax = plt.axes(projection='3d')
print(np.shape(hi[:, 1]))
print(hi)
ax.scatter3D(hi[:, 0], hi[:, 1], heights)
plt.show()