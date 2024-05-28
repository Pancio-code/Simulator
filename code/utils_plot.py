#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import sqlite3
import math

class UtilsPlot:

    @staticmethod
    def plot_data_average_client_fps_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), count(*)
from (
	select 
		id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
            ''')

        sum_reward = 0.0
        added = 0

        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_average_lag_over_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), avg(time_total)-{1/max_fps:.3f}
from (
	select 
		id, time_total, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
''')

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = max(0, line[1])

            sum_reward += reward*1000
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_average_response_time_over_time(node_uid, db_path, job_type=0, average_every_secs=15, max_fps=30):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f'''
select 
	cast(finish_time as int), avg(time_total)
from (
	select 
		id, time_total, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
	from 
		jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0
	) 
where lag_time > 0
group by cast(finish_time as int)
            ''')
        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1]  # = min(max_fps, line[1])

            sum_reward += reward*1000
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_reward_over_time(node_uid, db_path, average_every_secs=15):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f"select cast(generated_at as integer), sum(reward) from jobs where node_uid = {node_uid} group by cast(generated_at as integer)")

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_data_in_deadline_over_time(node_uid, db_path, average_every_secs=15):
        db = sqlite3.connect(db_path)
        cur = db.cursor()
        x_rewards = []
        y_rewards = []

        res = cur.execute(
            f"select cast(generated_at as integer), 1.0-(cast(sum(over_deadline) as float)/count(*)) from jobs where node_uid = {node_uid} and executed = 1 and rejected = 0 group by cast(generated_at as integer)")

        sum_reward = 0.0
        added = 0
        for line in res:
            t = line[0]
            reward = line[1]

            sum_reward += reward
            added += 1

            if t % average_every_secs == 0 and t > 0:
                # print(f"t={t}, added={added}, avg={sum_reward / added}")
                x_rewards.append(t)
                y_rewards.append(sum_reward / added)
                added = 0
                sum_reward = 0.0

        cur.close()
        db.close()

        return x_rewards, y_rewards

    @staticmethod
    def plot_actions_over_time(scheduler_node_uid, db_path, average_every_secs=15, n_clusters=3, n_workers=2):
        x_actions = []
        y_actions = []

        db = sqlite3.connect(db_path)
        cur = db.cursor()

        res = cur.execute("select max(generated_at) from jobs")
        for line in res:
            simulation_time = math.ceil(line[0])

        # compute jobs per second
        res = cur.execute(
            f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} group by cast(generated_at as integer)")

        jobs_per_second = {}
        for line in res:
            jobs_per_second[line[0]] = line[1]

        # compute how many actions per node in seconds
        actions = []

        # num of jobs
        num_d = {}
        res = cur.execute(
            f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} group by cast(generated_at as integer)")
        for line in res:
            num_d[line[0]] = line[1]

        # reject
        reject_d = {}
        res = cur.execute(
            f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} and action = 0 group by cast(generated_at as integer)")
        for line in res:
            reject_d[line[0]] = 100 * line[1] / num_d[line[0]]
        actions.append(reject_d)

        # cloud
        cloud_d = {}
        res = cur.execute(
            f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} and action = 1 group by cast(generated_at as integer)")
        for line in res:
            cloud_d[line[0]] = 100 * line[1] / num_d[line[0]]
        actions.append(cloud_d)

        # workers
        for node_i in range(2, n_workers + 2):
            actions_d = {}
            res = cur.execute(
                f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} and action = {node_i} group by cast(generated_at as integer)")

            for line in res:
                actions_d[line[0]] = 100 * line[1] / num_d[line[0]]
            actions.append(actions_d)

        # clusters
        for cluster_i in range(n_workers + 2, n_workers + n_clusters + 2 - 1): # exclude the current cluster
            actions_d = {}
            res = cur.execute(
                f"select cast(generated_at as integer), count(*) from jobs where node_uid = {scheduler_node_uid} and action = {cluster_i} group by cast(generated_at as integer)")

            for line in res:
                actions_d[line[0]] = 100 * line[1] / num_d[line[0]]
            actions.append(actions_d)

        out_x = []
        out_y = []

        for i in range(len(actions)):
            out_x.append(actions[i].keys())
            out_y.append(actions[i].values())

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

        return actions_avg_x, actions_avg_y
