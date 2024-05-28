#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import os
import pickle
import sqlite3
import time
from typing import List

from job import Job
from log import Log
from node import Node

MODULE = "ServiceDataStorage"

BASE_LOG_DIR = "_log"
PATH_RESULTS = "../results"
PATH_RESULTS_DATA = f"{PATH_RESULTS}/data"
os.makedirs(PATH_RESULTS_DATA, exist_ok=True)

# noinspection SqlNoDataSourceInspection
class ServiceDataStorage:

    def __init__(self, nodes: List[Node], session_id: str, learning_type, no_learning_policy, action_space):
        self._nodes = nodes
        self._n_nodes = len(nodes)
        self._session_id = session_id
        self._learning_type = learning_type

        # init dirs and db
        if learning_type == Node.LearningType.NO_LEARNING:
            self._log_dir = f"{PATH_RESULTS_DATA}/{BASE_LOG_DIR}/no-learning/{no_learning_policy.name}/{session_id}"
        else:
            self._log_dir = f"{PATH_RESULTS_DATA}/{BASE_LOG_DIR}/learning/{learning_type.name}/{action_space.name}/{session_id}"

        os.makedirs(self._log_dir, exist_ok=True)

        self._db = sqlite3.connect(':memory:')  # f"{self._log_dir}/log.db")
        self._db_cur = self._db.cursor()

        self._init_db()

        self._log_filename = f"{self._log_dir}/log.txt"
        self._log_meta_filename = f"{self._log_dir}/meta.txt"

        self._counter_rt_jobs_executed = [0 for _ in range(self._n_nodes)]
        self._counter_rt_jobs_executed_overdeadline = [0 for _ in range(self._n_nodes)]
        self._counter_rt_jobs_rejected = [0 for _ in range(self._n_nodes)]

        self._counter_nrt_jobs_executed = [0 for _ in range(self._n_nodes)]
        self._counter_nrt_jobs_rejected = [0 for _ in range(self._n_nodes)]

        self._rewards = [0 for _ in range(self._n_nodes)]

        self._counter_total_jobs = 0

    def _init_db(self):
        self._db_cur.execute('''CREATE TABLE round (
                                                    time real, 
                                                    worker_id integer,
                                                    battery_residual real,
                                                    variance real
                                                )''')

        self._db_cur.execute('''CREATE TABLE end_batteries (
                                                    time real, 
                                                    worker_id integer,
                                                    max_battery real
                                                )''')

        self._db_cur.execute('''CREATE TABLE episodes (
                                                    episode integer, 
                                                    node_uid integer, 
                                                    eps real, 
                                                    score real, 
                                                    total_jobs integer, 
                                                    loss real, 
                                                    mse real, 
                                                    mae real
                                                )''')

        self._db_cur.execute('''CREATE TABLE jobs (
                                                    id text, 
                                                    node_uid integer, 
                                                    episode integer, 
                                                    eps real, 
                                                    state_snapshot text,
                                                    forwarded_to_node_uid integer,
                                                    forwarded_to_cluster_uid integer,
                                                    forwarded_to_cloud integer,
                                                    action integer, 
                                                    executed integer, 
                                                    rejected integer, 
                                                    type integer, 
                                                    over_deadline integer, 
                                                    done integer, 
                                                    reward real, 
                                                    time_total real, 
                                                    time_probing real, 
                                                    time_dispatching real, 
                                                    time_queue real,
                                                    time_execution real,
                                                    time_total_execution real,
                                                    generated_at real
                                                )''')

        self._db_cur.execute('''CREATE TABLE q_values (
                                                    state text,
                                                    node_uid integer,
                                                    episode integer,
                                                    action integer,
                                                    value real,
                                                    primary key (node_uid, state, action)
                                                )''')

        self._db_cur.execute('''CREATE TABLE q_values_by_time (
                                                    time integer,
                                                    state text,
                                                    node_uid integer,
                                                    action integer,
                                                    value real,
                                                    primary key (time, node_uid, state, action)
                                                )''')
        self._db.commit()
        Log.minfo(MODULE, "DB init")

    def _copy_db_to_file(self):
        Log.minfo(MODULE, "Copying memory db to file, please wait")
        start = time.time()

        new_db = sqlite3.connect(f"{self._log_dir}/log.db")
        query = "".join(line for line in self._db.iterdump())

        # Dump old database in the new one.
        new_db.executescript(query)
        new_db.close()

        Log.minfo(MODULE, f"Done in {time.time() - start:2f}")

    def add_line_to_log(self, line):
        logfile = open(self._log_filename, "a")
        logfile.write(line + "\n")
        logfile.close()

    def _save_models(self):
        for node in self._nodes:
            Log.minfo(MODULE, f"Saving model for Node {node.get_uid()}")
            if node.get_learning_type() == Node.LearningType.D_SARSA and node.get_type() == Node.NodeType.SCHEDULER:
                fn = node.get_value_function()
                model_f = open(f"{node.get_models_dir()}/d_sarsa.model", "wb")
                pickle.dump(fn, model_f)
                model_f.close()
            else:
                Log.minfo(MODULE, "Skipped..")

    #
    # Stat data manager
    #

    def done_episode(self, node_uid, episode, eps, score, total_jobs, loss, mse, mae):
        # noinspection SqlResolve
        self._db_cur.execute(
            f'''INSERT INTO episodes VALUES ({episode}, {node_uid}, {eps},{score}, {total_jobs}, {loss}, {mse}, {mae})''')
        self._db.commit()


    def done_job(self, job: Job, reward: int):
        # noinspection SqlResolve
        self._db_cur.execute(f'''INSERT INTO jobs VALUES (
                                    "{job.get_uid()}", 
                                    {job.get_originator_node_uid()}, 
                                    {job.get_episode()}, 
                                    {job.get_eps()},
                                    "{job.get_state_snapshot_str()}",
                                    {job.get_forwarded_to_node_id()},
                                    {job.get_forwarded_to_cluster_id()},
                                    {1 if job.is_forwarded_to_cloud() else 0},
                                    {job.get_action(0)},
                                    {1 if job.is_executed() else 0}, 
                                    {1 if job.is_rejected() else 0},
                                    {job.get_type()},
                                    {1 if job.is_over_deadline() else 0},
                                    {1 if job.is_done() else 0},
                                    {reward}, 
                                    {job.get_total_time()},
                                    {job.get_probing_time()},
                                    {job.get_dispatched_time()},
                                    {job.get_queue_time()},
                                    {job.get_time_execution()},
                                    {job.get_total_time_execution()},
                                    {job.get_generated_at()})
                                ''')
        self._db.commit()

        # save to files
        self._rewards[job.get_originator_node_uid()] += reward
        self._counter_total_jobs += 1

        if job.get_episode() % 100 == 0:
            self.print_data(only_to_file=True)

    def log_q_value(self, state, node_uid, episode, action, value):
        # noinspection SqlResolve
        self._db_cur.execute(f'''REPLACE INTO q_values VALUES ("{state}", {node_uid}, {episode}, {action}, {value})''')
        self._db.commit()

    def log_q_value_at_time(self, time: int, state, node_uid, action, value):
        # noinspection SqlResolve
        self._db_cur.execute(f'''REPLACE INTO q_values_by_time VALUES (
                                    {time}, "{state}", {node_uid}, {action}, {value})''')
        self._db.commit()

    def done_simulation(self):
        self._copy_db_to_file()
        self._save_models()
        self._db.close()

    def print_data(self, only_to_file=False):
        total_rt_executed = sum(self._counter_rt_jobs_executed)
        total_rt_executed_overdeadline = sum(self._counter_rt_jobs_executed_overdeadline)
        total_rt_rejected = sum(self._counter_rt_jobs_rejected)
        total_nrt_executed = sum(self._counter_nrt_jobs_executed)
        total_nrt_rejected = sum(self._counter_nrt_jobs_rejected)
        total_reward = sum(self._rewards)
        total_rejected = total_rt_rejected + total_nrt_rejected

        log_meta_file = open(self._log_meta_filename, "w")
        print(self._counter_total_jobs, file=log_meta_file)
        print("total_rt_executed=%d" % total_rt_executed, file=log_meta_file)
        print("total_rt_executed_overdeadline=%d" % total_rt_executed_overdeadline, file=log_meta_file)
        if total_rt_executed > 0:
            print("total_rt_executed_overdeadline_perc=%.2f" % (total_rt_executed_overdeadline / total_rt_executed),
                  file=log_meta_file)
        print("total_rt_rejected=%d" % total_rt_rejected, file=log_meta_file)
        print("total_nrt_executed=%d" % total_nrt_executed, file=log_meta_file)
        print("total_nrt_rejected=%d" % total_nrt_rejected, file=log_meta_file)
        print("total_jobs=%d" % self._counter_total_jobs, file=log_meta_file)
        print("total_rejected=%d" % total_rejected, file=log_meta_file)
        print("total_reward=%d" % total_reward, file=log_meta_file)
        if self._counter_total_jobs > 0:
            print("total_rejected_perc=%.2f" % (total_rejected / self._counter_total_jobs), file=log_meta_file)
            print("total_reward / total_jobs=%.2f" % (total_reward / self._counter_total_jobs), file=log_meta_file)
        log_meta_file.close()

        if not only_to_file:
            print()
            print("### DataStorage report ###")
            print("total_rt_executed=%d" % total_rt_executed)
            print("total_rt_executed_overdeadline=%d" % total_rt_executed_overdeadline)
            print("total_rt_executed_overdeadline_perc=%.2f" % (total_rt_executed_overdeadline / total_rt_executed))
            print("total_rt_rejected=%d" % total_rt_rejected)
            print("total_nrt_executed=%d" % total_nrt_executed)
            print("total_nrt_rejected=%d" % total_nrt_rejected)
            print("self._counter_total_jobs=%d" % self._counter_total_jobs)
            print("total_rejected=%d" % total_rejected)
            print("total_rejected_perc=%.2f" % (total_rejected / self._counter_total_jobs))

    def get_log_dir(self):
        return self._log_dir

    def log_battery(self, timestamp, worker_id, battery_residual,variance):
        self._db_cur.execute(
            f'''INSERT INTO round VALUES (
                        {timestamp}, 
                        {worker_id},
                        {battery_residual},
                        {variance}
            )''')
        self._db.commit()

    def log_end_battery(self, timestamp, worker_id, max_battery):
        self._db_cur.execute(
            f'''INSERT INTO end_batteries VALUES (
                        {timestamp}, 
                        {worker_id},
                        {max_battery}
            )''')
        self._db.commit()
