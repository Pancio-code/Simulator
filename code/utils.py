#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import sqlite3
from datetime import datetime


class Utils:

    @staticmethod
    def current_time_string():
        return datetime.now().strftime("%Y%m%d-%H%M%S")

    @staticmethod
    def compute_average_client_fps(node_uid, db_path, job_type=0, from_time=9000, to_time=10000):
        db = sqlite3.connect(db_path)
        cur = db.cursor()

        res = cur.execute(
            f'''
select 
	avg(fps)
from (
	select 
		cast(finish_time as int), count(*) fps
	from (
		select 
			id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
		from 
			jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0 and generated_at >= 9000 and generated_at < 10000
		) 
	where lag_time > 0
	group by cast(finish_time as int)
	)
            ''')

        for line in res:
            return line[0]

    @staticmethod
    def compute_average_response_time(node_uid, db_path, job_type=0, from_time=9000, to_time=10000):
        db = sqlite3.connect(db_path)
        cur = db.cursor()

        res = cur.execute(
            f'''
        select 
	avg(avg_time_total)
from (
	select 
		cast(finish_time as int), avg(time_total) avg_time_total
	from (
		select 
			id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time, time_total 
		from 
			jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0 and generated_at >= 9000 and generated_at < 10000
		) 
	where lag_time > 0
	group by cast(finish_time as int)
)           ''')

        for line in res:
            return line[0]

    @staticmethod
    def compute_average_lag_time(node_uid, db_path, job_type=0, from_time=9000, to_time=10000):
        db = sqlite3.connect(db_path)
        cur = db.cursor()

        res = cur.execute(
            f'''
select 
	avg(avg_lag_time)
from (
	select 
		cast(finish_time as int), avg(lag_time) avg_lag_time
	from (
		select 
			id, generated_at+time_total as finish_time, (generated_at+time_total) - LAG(generated_at+time_total, 1) OVER (ORDER BY generated_at) as lag_time 
		from 
			jobs where node_uid = {node_uid} and type = {job_type} and executed = 1 and rejected = 0 and generated_at >= 9000 and generated_at < 10000
		) 
	where lag_time > 0
	group by cast(finish_time as int)
	)
            ''')

        for line in res:
            return line[0]