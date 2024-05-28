#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.
import os
import sqlite3

import matplotlib.pyplot as plt
import numpy as np

from plot import PlotUtils

N_NODES = 6
N_SLOTS = 4 * 24

MAX_LIMIT = 0.9
MIN_LIMIT = 0.1

PLOT_DIR = "data"
os.makedirs(PLOT_DIR, exist_ok=True)

db = sqlite3.connect("log_saved.db")
cur = db.cursor()

x_arr = []
y_arr = []
max_v = 0
min_v = 10000


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


for i in range(N_NODES):
    x = []
    y = []

    res = cur.execute(f'''select node_uid, slot, avg(value) from traffic where node_uid = {i} group by node_uid, slot order by slot''')
    for line in res:
        x.append(line[1])
        y.append(line[2])

    # smooth
    y = savitzky_golay(np.array(y), 17, 4)

    x_arr.append(x)
    y_arr.append(y)

# find min max
for i in range(N_NODES):
    for j in range(len(y_arr[i])):
        if y_arr[i][j] > max_v:
            max_v = y_arr[i][j]
        if y_arr[i][j] < min_v:
            min_v = y_arr[i][j]

print(f"max_v={max_v}, min_v={min_v}")

# normalize everything
for i in range(N_NODES):
    for j in range(len(y_arr[i])):
        y_arr[i][j] = (((y_arr[i][j] - min_v) / (max_v - min_v)) * (MAX_LIMIT - MIN_LIMIT)) + MIN_LIMIT

PlotUtils.use_tex()
plt.clf()
for i in range(len(x_arr)):
    plt.plot(x_arr[i], y_arr[i])

plt.legend([f"Node\#{i}" for i in range(N_NODES)])
plt.savefig(f"{PLOT_DIR}/traffic_node_comparison.pdf")
plt.ylabel(r"\rho")
plt.xlabel("Time Slot")

# plot singles
for i in range(N_NODES):
    plt.clf()
    plt.plot(x_arr[i], y_arr[i])
    plt.savefig(f"{PLOT_DIR}/traffic_node_{i}.pdf")

    outf = open(f"{PLOT_DIR}/traffic_node_{i}.csv", "w")
    for j in range(len(x_arr[i])):
        print(f"{x_arr[i][j]},{y_arr[i][j]}", file=outf)
    outf.close()


entropy = np.array(y_arr[0])
entropy = savitzky_golay(entropy, 47, 3)
plt.clf()
plt.plot(x_arr[0], entropy)
plt.show()