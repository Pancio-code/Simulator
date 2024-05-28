#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.
import datetime

import matplotlib.pyplot as plt
from polycircles import polycircles

from traffic.city.common import TimeSlots, NodePosition

FILENAME = "trip_data_1.csv"

MAX_LINES = 1000000

KEY_PICKUP_LAT = "pickup_latitude"
KEY_PICKUP_LON = "pickup_longitude"
KEY_DROPOFF_LAT = "dropoff_latitude"
KEY_DROPOFF_LON = "dropoff_longitude"
KEY_PICKUP_DATETIME = "pickup_datetime"
KEY_DROPOFF_DATETIME = "dropoff_datetime"

# nodes
nodes = [NodePosition(-74.00516, 40.74046),
         NodePosition(-73.99961, 40.73828),
         NodePosition(-73.99421, 40.73638),
         NodePosition(-73.98887, 40.73420),
         NodePosition(-73.98261, 40.73152),
         NodePosition(-73.97780, 40.72979)]

time_slots = TimeSlots(nodes_position=nodes)

cols = {}
lats_p = []
longs_p = []
lats_d = []
longs_d = []

csv_file = open(FILENAME, "r")
print("line: 0", end="")
for i, line in enumerate(csv_file):
    print(f"\rline: {i}", end="")

    if i == 0:
        cmps = line.split(",")
        for j, field in enumerate(cmps):
            cols[str.strip(field)] = j
        continue

    if i > MAX_LINES:
        break

    cmps = line.split(",")

    lat_p_str = cmps[cols[KEY_PICKUP_LAT]]
    lat_d_str = cmps[cols[KEY_DROPOFF_LAT]]
    long_p_str = cmps[cols[KEY_PICKUP_LON]]
    long_d_str = cmps[cols[KEY_DROPOFF_LON]]

    if lat_p_str == "" or lat_d_str == "" or long_p_str == "" or long_d_str == "":
        continue

    lat_p = float(cmps[cols[KEY_PICKUP_LAT]])
    long_p = float(cmps[cols[KEY_PICKUP_LON]])
    lat_d = float(cmps[cols[KEY_DROPOFF_LAT]])
    long_d = float(cmps[cols[KEY_DROPOFF_LON]])

    # check if coordinates are valid
    if not (-90 <= lat_p <= 90 and -90 <= lat_d <= 90 and -90 <= long_p <= 90 and -90 <= long_d <= 90
            and int(lat_p) != 0 and int(long_p) != 0 and int(lat_d) != 0 and int(long_d) != 0):
        continue

    # add to list
    lats_p.append(lat_p)
    longs_p.append(long_p)
    lats_d.append(lat_d)
    longs_d.append(long_d)

    pickup_datetime = datetime.datetime.strptime(cmps[cols[KEY_PICKUP_DATETIME]], '%Y-%m-%d %H:%M:%S')
    dropoff_datetime = datetime.datetime.strptime(cmps[cols[KEY_DROPOFF_DATETIME]], '%Y-%m-%d %H:%M:%S')

    # log the traffic
    time_slots.track_slots(pickup_datetime, dropoff_datetime, lat_p, long_p, lat_d, long_d)

print()
time_slots.save_db_to_file()

# print(lats)
# print(longs)

fig, ax = plt.subplots()
ax.scatter(longs_p, lats_p, s=2, facecolor='blue')
ax.scatter(longs_d, lats_d, s=2, facecolor='skyblue')

w = 6.4
h = 4.8
r = w / h
fig.set_figwidth(6.4)  # 6.4
fig.set_figheight(4.8)  # 4.8

x1 = -74.015
x2 = -73.97
y1 = 40.72
y2 = y1 + (x2 - x1) / r

plt.xlim(x1, x2)
plt.ylim(y1, y2)

print(f"fig_ratio={r}, y2={y2}")

# print nodes
nodes_x = []
nodes_y = []
for node in nodes:
    nodes_x.append(node.x)
    nodes_y.append(node.y)

ax.scatter(nodes_x, nodes_y, s=100, facecolor='red')
# ax.scatter(nodes_x, nodes_y, s=2000, facecolors='none', edgecolors='red')

areas_x = []
areas_y = []

for node in nodes:
    polycircle = polycircles.Polycircle(latitude=node.y, longitude=node.x, radius=500, number_of_vertices=360)
    for point in polycircle.to_lat_lon():
        areas_x.append(point[1])
        areas_y.append(point[0])

ax.scatter(areas_x, areas_y, s=1, facecolor='red')

plt.show()
