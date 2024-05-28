#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import math
import os
import sqlite3

from log import Log
from plot import Plot, PlotUtils
from utils import Utils
from datetime import datetime

MODULE = "PlotRewardOverTime"

N_NODES = 3

LEGEND = [
    "Reject",
    "Cloud",
    "Worker 1 (S = 1.0)",
    "Worker 2 (S = 0.9)",
    "Worker 3 (S = 0.6)",
]

alpha = "1.00"
type = "_FAILURE"
SESSION_ID = datetime.now().strftime("%Y%m%d")
PATH_RESULTS = "./results"
PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"

#llna_db_file = f"./_log/no-learning/LEAST_LOADED_NOT_AWARE/20240514_LEAST_LOADED_NOT_AWARE_ONLY_WORKERS/log.db"
llna_db_file = f"./code/_log/learning/D_SARSA/WORKERS_OR_CLOUD/{SESSION_ID}_{alpha}{type}/log.db"
print(llna_db_file)

db = sqlite3.connect(llna_db_file)
cur = db.cursor()

res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

Log.mdebug(MODULE, f"simulation_time={simulation_time}")

# compute jobs per second
res = cur.execute(
    "select cast(generated_at as integer), count(*) from jobs where node_uid = 0 group by cast(generated_at as integer)")

jobs_per_second = {}
for line in res:
    jobs_per_second[line[0]] = line[1]

# compute how many actions per node in seconds
actions = []

num_of_jobs_per_t = {}
# num of jobs
num_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs group by cast(generated_at as integer)")
for line in res:
    num_d[line[0]] = line[1]

# reject
reject_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = 0 group by cast(generated_at as integer)")
for line in res:
    reject_d[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(reject_d)

# cloud
cloud_d = {}
res = cur.execute(
    f"select cast(generated_at as integer), count(*) from jobs where action = 1 group by cast(generated_at as integer)")
for line in res:
    cloud_d[line[0]] = 100 * line[1] / num_d[line[0]]
actions.append(cloud_d)

for node_i in range(2, N_NODES + 2):
    actions_d = {}
    res = cur.execute(
        f"select cast(generated_at as integer), count(*) from jobs where action = {node_i} group by cast(generated_at as integer)")

    for line in res:
        actions_d[line[0]] = 100 * line[1] / num_d[line[0]]
    actions.append(actions_d)

os.makedirs(f"{PATH_RESULTS_PLOT}", exist_ok=True)
figure_filename = f"{PATH_RESULTS_PLOT}/actions-over-time_{SESSION_ID}_{alpha}{type}.pdf"

out_x = []
out_y = []

for i in range(len(actions)):
    out_x.append(actions[i].keys())
    out_y.append(actions[i].values())

# Plot.multi_plot(out_x, out_y, "Time", "Percentage of actions", fullpath=figure_filename, legend=LEGEND)

# create the average
actions_avg_x = []
actions_avg_y = []
average_every_secs = 300

for action_d in actions:
    x = []
    y = []

    sum_v = 0
    elements = 0

    for i in range(simulation_time):
        if i in action_d.keys():
            sum_v += action_d[i]
        elements += 1

        if i % average_every_secs == 0:
            x.append(i)
            y.append(sum_v / elements)

            sum_v = 0
            elements = 0

    actions_avg_x.append(x)
    actions_avg_y.append(y)

figure_filename = f"{PATH_RESULTS_PLOT}/actions-over-time-avg_{SESSION_ID}_{alpha}{type}.pdf"
Plot.multi_plot(actions_avg_x, actions_avg_y, "Time (s)", "Percentage of actions ", fullpath=figure_filename,
                legend=LEGEND, figh=2.7, legend_position='upper right', use_grid=True, filename=figure_filename)
