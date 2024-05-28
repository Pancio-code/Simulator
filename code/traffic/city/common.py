#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import datetime
import os
import sqlite3
import time
from typing import List

from geographiclib import geodesic


class NodePosition:
    def __init__(self, x, y):
        self.x = x  # long
        self.y = y  # lat

    def dist_from(self, lat, long):
        return geodesic.Geodesic.WGS84.Inverse(self.y, self.x, lat, long)['s12']


# noinspection SqlResolve
class TimeSlots:
    MODULE = "TimeSlots"

    def __init__(self, nodes_position=None, n_slots=96, max_node_distance_m=1000):
        """Init with the number of slots during the day"""
        self._n_slots = n_slots
        self._slots = [0 for _ in range(n_slots)]
        self._nodes_position = nodes_position  # type: List[NodePosition]

        self._day_seconds = 24 * 3600
        self._day_slot_size_seconds = self._day_seconds / self._n_slots

        self._max_node_distance_m = max_node_distance_m

        self._db = sqlite3.connect(":memory:")
        self._cur = self._db.cursor()

        self._cur.execute('''CREATE TABLE traffic(
                                node_uid integer,
                                day text,
                                slot integer,
                                value integer,
                                primary key (node_uid, day, slot)
                                )''')

    def track_slots(self, start_time, end_time, start_lat, start_long, end_lat, end_long, speed_kmh=60):
        """Log the slot to db every step_secs"""

        # compute distance between start and end
        dist = geodesic.Geodesic.WGS84.Inverse(start_lat, start_long, end_lat, end_long)
        dist_m = dist['s12']
        dist_azi = dist['azi1']
        # print(f"track_slots: d(({start_lat},{start_long}), ({end_lat}, {end_long}))={dist_m}")

        # compute step_sec
        total_transit_time = end_time.timestamp() - start_time.timestamp()
        # print(f"track_slots: total_transit_time={total_transit_time}s ({total_transit_time / 3600}hrs)")

        number_of_slots = int(total_transit_time / self._day_slot_size_seconds) + 1
        # print(f"track_slots: number_of_slots={number_of_slots}")

        dist_step_m = dist_m / number_of_slots
        # print(f"track_slots: dist_step={dist_step_m}")

        current_secs = start_time.timestamp()
        start_time_slot = self.get_slot_from_ts(current_secs)
        secs_to_end_slot = current_secs - start_time_slot * self._day_slot_size_seconds
        # print(f"track_slots: start_time_slot={start_time_slot}, secs_to_end_slot={secs_to_end_slot}")

        for i in range(number_of_slots):
            step = geodesic.Geodesic.WGS84.Direct(start_lat, start_long, dist_azi, i * dist_step_m)
            step_lat2 = step['lat2']
            step_lon2 = step['lon2']

            # print(f"track_slots: point #{i}, lat={step_lat2}, lon={step_lon2}, dist={i * dist_step_m}, "
            #       f"current_secs={current_secs}s, elapsed={(current_secs - start_time.timestamp()) / 3600}h")

            nearest_node = self.pick_nearest_node(step_lat2, step_lon2)
            if nearest_node > -1:
                self.add_to_slot_from_ts(nearest_node, current_secs)

            current_secs += (dist_step_m / (speed_kmh / 3.6))

        # print()

    def get_slot_from_ts(self, ts):
        """Retrieve the time slot from the timestamp"""
        date = datetime.datetime.fromtimestamp(ts)
        ts_midnight = date.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        ts_seconds = ts - ts_midnight

        return int(ts_seconds / self._day_slot_size_seconds)

    def add_to_slot_from_time(self, node_uid, time):
        """Computes where the datetime passed falls in which day slot and add it to db"""
        ts = time.timestamp()
        ts_midnight = time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        ts_seconds = ts - ts_midnight

        slot = int(ts_seconds / self._day_slot_size_seconds)

        # self._slots[slot] += 1
        self.inc_entry_to_db(node_uid, time.day, time.month, time.year, slot)

        # print(f"ts={ts}, ts_new={ts_midnight}, ts_seconds={ts_seconds}, slot={slot}")
        # print(f"day={time.day}, month={time.month}, year={time.year}")

        return slot

    def add_to_slot_from_ts(self, node_uid, ts):
        """Increase the entry by one in the slot given the timestamp"""
        date = datetime.datetime.fromtimestamp(ts)
        self.add_to_slot_from_time(node_uid, date)

    def inc_entry_to_db(self, node_uid, day, month, year, slot):
        """Increase the counter for a item in db"""
        day_str = f"{day}-{month}-{year}"

        # retrieve the value if one
        res = self._cur.execute(f'''select * from traffic where node_uid = {node_uid} and day = "{day_str}" and slot = {slot}''')
        res_v = 0
        for line in res:
            res_v = line[3]  # pick the value

        # add to db
        self._cur.execute(f'''replace into traffic values({node_uid}, "{day_str}", {slot}, {res_v + 1})''')

    def pick_nearest_node(self, lat, long):
        min_i = -1
        min_d = self._max_node_distance_m

        for i, node in enumerate(self._nodes_position):
            dist_m = node.dist_from(lat, long)
            if dist_m <= self._max_node_distance_m:
                if dist_m < min_d:
                    min_d = dist_m
                    min_i = i

        return min_i

    def save_db_to_file(self):
        os.remove("./log.db")

        print("Copying memory db to file, please wait")
        start = time.time()

        new_db = sqlite3.connect(f"./log.db")
        query = "".join(line for line in self._db.iterdump())

        # Dump old database in the new one.
        new_db.executescript(query)
        new_db.close()

        print(f"Done in {time.time() - start:2f}")


'''
ts = TimeSlots()
print(ts.add_to_slot_from_time(0, datetime.datetime.now()))
print(ts.add_to_slot_from_time(0, datetime.datetime.now()))
print(ts.add_to_slot_from_time(0, datetime.datetime.now()))
print(ts.add_to_slot_from_time(0, datetime.datetime.now()))

start_time = datetime.datetime.now().replace(hour=6)
end_time = datetime.datetime.now()

print(f"start_time={start_time}")
print(f"end_time={end_time}")

print(f"total_slots={ts.track_slots(0, start_time, end_time, 42.02130219175812, 12.91242670668925, 41.86665766387956, 12.580559525671836)}")
ts.save_db_to_file()
'''
