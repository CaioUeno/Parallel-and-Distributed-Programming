import matplotlib.pyplot as plt
import numpy as np

instances = open("instances.txt", "r")
lines = [list(map(float, line.split())) for line in instances.readlines()]
lines = np.array(lines)

labels = open("labels.txt", "r")
labels = [list(map(int, line.split())) for line in labels.readlines()]
labels = np.array(labels).ravel()

color_map = {0:'r', 1:'g', 2:'b', 3:'y'}
colors = [color_map[label] for label in labels]

plt.scatter(lines[:, 0], lines[:, 1], color=colors)
plt.show()
