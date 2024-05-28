#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.
import math
import os
import sqlite3

from matplotlib import pyplot as plt
from datetime import datetime
from plot import PlotUtils, Plot
from utils import Utils
from utils_plot import UtilsPlot

LEGEND = ["WORKER 1.0", "WORKER 0.9", "WORKER 0.6"]
cmap_def = plt.get_cmap("tab10")


alpha = "0.00"
type = "_FAILURE"
SESSION_ID = datetime.now().strftime("%Y%m%d")
PATH_RESULTS = "./results"
PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"

#llna_db_file = f"./_log/no-learning/LEAST_LOADED_NOT_AWARE/20240514_LEAST_LOADED_NOT_AWARE_ONLY_WORKERS/log.db"
llna_db_file = f"./code/_log/learning/D_SARSA/WORKERS_OR_CLOUD/{SESSION_ID}_{alpha}{type}/log.db"
print(llna_db_file)

db = sqlite3.connect(llna_db_file)
cur = db.cursor()

cur.execute("SELECT time, worker_id, battery_residual, variance FROM round")
data = cur.fetchall()

# Organize the data
worker_data = {}
for time, worker_id, battery_residual, variance in data:
    if worker_id not in worker_data:
        worker_data[worker_id] = []
    
    worker_data[worker_id].append((time, battery_residual, variance))

print(worker_data.keys())

# Plot the battery levels
plt.figure()
for worker_id, battery_data in worker_data.items():
    timestamps, battery_levels, _ = zip(*battery_data)
    plt.plot(timestamps, battery_levels, label=f'Worker {worker_id} Battery')

plt.xlabel('Timestamp')
plt.ylabel('Battery Residual')
plt.legend()

os.makedirs(PATH_RESULTS_PLOT, exist_ok=True)
figure_filename = f"{PATH_RESULTS_PLOT}/battery_levels_{SESSION_ID}_{alpha}{type}.pdf"

plt.savefig(figure_filename)
plt.close()

# Plot the variances
plt.figure()
for worker_id, battery_data in worker_data.items():
    timestamps, _, variances = zip(*battery_data)
    plt.plot(timestamps, variances, label=f'Worker {worker_id} Variance')

plt.xlabel('Timestamp')
plt.ylabel('Variance')
plt.legend()

figure_filename = f"{PATH_RESULTS_PLOT}/variances_{SESSION_ID}_{alpha}{type}.pdf"
plt.savefig(figure_filename)
plt.close()

average_every_secs = 120

LEGEND = ["60FPS Client", "30FPS Client", "15FPS Client"]
FPS_LIMITS_MAX = [60, 30, 15]
FPS_LIMITS_MIN = [50, 20, 10]
cmap_def = plt.get_cmap("tab10")


db = sqlite3.connect(llna_db_file)
cur = db.cursor()

simulation_time = 0
res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

cur.close()
db.close()

x_arr = []
y_arr = []

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=0, max_fps=FPS_LIMITS_MAX[0], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=1, max_fps=FPS_LIMITS_MAX[1], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)

x, y = UtilsPlot.plot_data_average_client_fps_time(0, llna_db_file, job_type=2, max_fps=FPS_LIMITS_MAX[2], average_every_secs=average_every_secs)
x_arr.append(x)
y_arr.append(y)


figure_filename = f"{PATH_RESULTS_PLOT}/average-client-fps_{SESSION_ID}_{alpha}{type}.pdf"


plt.clf()
fig, ax = plt.subplots()
markers = ["^", "*", "s", "o", "g"]

legend_arr = []

for i in range(len(y_arr)):
    line, = ax.plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                    marker=markers[i % len(markers)],
                    markersize=5, markeredgewidth=0.6)
    print(i)
    # plt.hlines(FPS_LIMITS_MAX[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    # plt.hlines(FPS_LIMITS_MIN[i], 0, simulation_time, colors=cmap_def(i), linestyles='solid', label=LEGEND[i], linewidth=0.6)
    plt.fill_between(x, FPS_LIMITS_MAX[i], FPS_LIMITS_MIN[i], color=cmap_def(i), alpha=0.3, linewidth=0)

    legend_arr.append(line)

plt.legend(legend_arr, LEGEND, fontsize="small")  # , loc="lower right")

ax.set_xlabel("Time (s)")
ax.set_ylabel("FPS")
ax.set_ylim([0,max(FPS_LIMITS_MAX)])
fig.tight_layout()


os.makedirs(f"{PATH_RESULTS_PLOT}", exist_ok=True)

plt.savefig(figure_filename)

plt.close(fig)