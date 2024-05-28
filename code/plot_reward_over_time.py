#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import math
import os
import sqlite3

from matplotlib import pyplot as plt
from datetime import datetime
from log import Log
from plot import PlotUtils
from utils import Utils
from utils_plot import UtilsPlot

MODULE = "PlotRewardOverTime"


alpha = "1.00"
type = "_FAILURE"
SESSION_ID = datetime.now().strftime("%Y%m%d")
PATH_RESULTS = "./results"
PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"

llna_db_file = f"./code/_log/learning/D_SARSA/WORKERS_OR_CLOUD/{SESSION_ID}_{alpha}{type}/log.db"
print(llna_db_file)

db = sqlite3.connect(llna_db_file)
cur = db.cursor()

res = cur.execute("select max(generated_at) from jobs")
for line in res:
    simulation_time = math.ceil(line[0])

Log.mdebug(MODULE, f"simulation_time={simulation_time}")

average_every_secs = 240

x_rewards, y_rewards = UtilsPlot.plot_data_reward_over_time(0, llna_db_file, average_every_secs=average_every_secs)

print(x_rewards)
print(y_rewards)

x_eps = []
y_eps = []

# eps
res = cur.execute(
    f"select cast(generated_at as integer), avg(eps) from jobs where node_uid = 0 group by cast(generated_at as integer)")
sum_reward = 0.0
added = 0
for line in res:
    t = line[0]
    reward = line[1]

    sum_reward += reward
    added += 1

    if t % average_every_secs == 0 and t > 0:
        # print(f"t={t}, added={added}, avg={sum_reward / added}")
        x_eps.append(t)
        y_eps.append(sum_reward / added)
        added = 0
        sum_reward = 0.0

os.makedirs(f"{PATH_RESULTS_PLOT}", exist_ok=True)
figure_filename = f"{PATH_RESULTS_PLOT}/reward-over-time_{SESSION_ID}_{alpha}{type}.pdf"

print(x_eps)
print(y_eps)

cmap_def = plt.get_cmap("tab10")

fig, ax = plt.subplots()
# make a plot
ax.plot(x_rewards, y_rewards, marker="o", markersize=3.0, markeredgewidth=1, linewidth=0.7,
        color=cmap_def(0))
ax.set_xlabel("Time")
ax.set_ylabel("Reward")

ax2 = ax.twinx()
ax2.plot(x_eps, y_eps, marker=None, markersize=3.0, markeredgewidth=1, linewidth=1, color=cmap_def(1))
ax2.set_xlabel("Time")
ax2.set_ylabel("epsilon")

fig.tight_layout(h_pad=0)
fig.set_figwidth(6.4)  # 6.4
fig.set_figheight(3.1)  # 4.8

plt.savefig(figure_filename)
