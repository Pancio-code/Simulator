#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.
import math
import os
import sqlite3

from matplotlib import pyplot as plt

from log import Log
from plot import PlotUtils
from utils import Utils
from utils_plot import UtilsPlot
from datetime import datetime

MODULE = "PlotRewardOverTime"

def plot_stacked(session_id, alpha, type, path_results_plot, path_results_data):
    llna_db_file = f"{path_results_data}/_log/learning/D_SARSA/WORKERS_OR_CLOUD/{session_id}_{alpha}{type}/log.db"
    print(llna_db_file)

    db = sqlite3.connect(llna_db_file)
    cur = db.cursor()

    res = cur.execute("select max(generated_at) from jobs")
    for line in res:
        simulation_time = math.ceil(line[0])

    Log.mdebug(MODULE, f"simulation_time={simulation_time}")

    average_every_secs = 250

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
            x_eps.append(t)
            y_eps.append(sum_reward / added)
            added = 0
            sum_reward = 0.0

    print(x_eps)
    print(y_eps)

    cmap_def = plt.get_cmap("tab10")
    PlotUtils.use_tex()

    fig, ax = plt.subplots(nrows=4, ncols=1, gridspec_kw={'height_ratios': [1, 2, 2, 2]})

    # make a plot
    ax[0].plot(x_rewards, y_rewards, marker=r"$\triangle$", markersize=3.0, markeredgewidth=1, linewidth=0.7,
            color=cmap_def(0))
    ax[0].set_ylabel(r"$R(\pi)$")

    ax2 = ax[0].twinx()
    ax2.plot(x_eps, y_eps, marker=None, markersize=3.0, markeredgewidth=1, linewidth=1, color=cmap_def(1))
    ax2.set_xlabel("Time")
    ax2.set_ylabel(r"$\epsilon$")
    ax[0].set_xlim([0, simulation_time])

    #
    # client fps
    #

    LEGEND = ["60FPS Client (Min. 50)", "30FPS Client (Min. 20)", "15FPS Client (Min. 10)"]
    FPS_LIMITS_MAX = [60, 30, 15]
    FPS_LIMITS_MIN = [50, 20, 10]

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


    markers = [r"$\triangle$", r"$\square$", r"$\diamondsuit$", r"$\otimes$", r"$\star$"]

    legend_arr = []

    for i in range(len(y_arr)):
        line, = ax[1].plot(x_arr[i], y_arr[i], markerfacecolor='None', linewidth=0.6,
                        marker=markers[i % len(markers)],
                        markersize=3, markeredgewidth=0.6)
        ax[1].fill_between(x, FPS_LIMITS_MAX[i], FPS_LIMITS_MIN[i], color=cmap_def(i), alpha=0.3, linewidth=0)

        legend_arr.append(line)

    ax[1].legend(legend_arr, LEGEND, fontsize="small",loc="lower left")

    ax[1].set_ylabel(r"$\omega_e$ (fps)")
    ax[1].set_ylim([0, max(FPS_LIMITS_MAX)])
    ax[1].set_xlim([0, simulation_time])

    #
    # battery
    #

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


    for worker_id, battery_data in worker_data.items():
        timestamps, battery_levels, _ = zip(*battery_data)
        line, = ax[2].plot(timestamps, battery_levels, label=f'Worker {worker_id} Battery',markerfacecolor='None', linewidth=0.6,
                        marker=markers[i % len(markers)],
                        markersize=1, markeredgewidth=0.6)

    ax[2].set_ylabel("B (wh)")
    ax[2].set_xlim([0, simulation_time])

    #
    # Variance
    #

    for worker_id, battery_data in worker_data.items():
        timestamps, _, variances = zip(*battery_data)
        line, = ax[3].plot(timestamps, variances, label=f'Worker {worker_id} Variance', markerfacecolor='None', linewidth=0.6,
                        marker=markers[i % len(markers)],
                        markersize=1, markeredgewidth=0.6)

    ax[3].set_ylabel(r"$\sigma$")
    ax[3].set_xlim([0, simulation_time])

    #
    # final
    #

    ax[3].set_xlabel("Time (s)")

    fig.tight_layout(h_pad=0, w_pad=0)
    fig.set_figwidth(6.4)  # 6.4
    fig.set_figheight(7.5)  # 4.8

    os.makedirs(f"{path_results_plot}/", exist_ok=True)
    figure_filename = f"{path_results_plot}/plot_stacked_reward_fps_batteries_variance_{session_id}_{alpha}{type}.pdf"

    fig.subplots_adjust(
        top=0.979,
        bottom=0.067,
        left=0.095,
        right=0.917,
        hspace=0.15,
        wspace=0
    )

    plt.savefig(figure_filename, bbox_inches='tight', transparent="True", pad_inches=0)

if __name__ == "__main__":
    alpha = "0.00"
    type = "_FAILURE"
    SESSION_ID = datetime.now().strftime("%Y%m%d")
    PATH_RESULTS = "./results"
    PATH_RESULTS_PLOT = f"{PATH_RESULTS}/plot"
    PATH_RESULTS_DATA = f"./"

    plot_stacked(SESSION_ID, alpha, type, PATH_RESULTS_PLOT, PATH_RESULTS_DATA)

