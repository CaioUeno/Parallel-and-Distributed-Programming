import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import random

instances = open("instances.txt", "r")
lines = [list(map(float, line.split())) for line in instances.readlines()]
lines = np.array(lines)

labels = open("labels.txt", "r")
labels = [list(map(int, line.split())) for line in labels.readlines()]
labels = np.array(labels).ravel()

color_list = list(mcolors.CSS4_COLORS.keys())
random.shuffle(color_list)
color_map = {i:color for i, color in enumerate(color_list)}
colors = [color_map[label] for label in labels]

plt.scatter(lines[:, 0], lines[:, 1], color=colors)
plt.grid(True)
plt.show()
