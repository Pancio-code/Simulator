#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import shapefile
from mpl_toolkits.basemap import Basemap
from polycircles import polycircles

from traffic.city.common import NodePosition

title = 'nodes_position_basemap'
imgfile = './nodes_position_study/{}.pdf'.format(title.lower().replace(" ", "_"))
shpfile = "/home/gpm/Coding/papers/paper-2019-unk-deadline/simulators/ContinuousTimeDistributedMultiPowerOfNDeadlines/traffic/city/new_york_new_york_osm_roads/new_york_new_york_osm_roads"
fontcolor = '#666666'

fig = plt.figure()
ax = fig.add_subplot(111, frame_on=False)
# fig.suptitle(title, fontsize=50, y=.94, color=fontcolor)

sf = shapefile.Reader(shpfile)

x0, y0, x1, y1 = sf.bbox
cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

print(f"sf.bbox={sf.bbox}")
print(f"cx={cx}, cy={cy}")

# my computations
w = 6.4  # 6.4
h = 4.4  # 4.8
r = w / h
fig.set_figwidth(w)
fig.set_figheight(h)

x0 = -74.015
y0 = 40.72
x1 = -73.97

w_new = x1 - x0
h_new = w_new / r

y1 = y0 + h_new

cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

print(f"w/h={w / h}, w_new={w_new}, h_new={h_new}")
print(f"x0,y0={x0},{y0}, x1,y1={x1},{y1}")
print(f"cx={cx}, cy={cy}")

cmap_def = plt.get_cmap("tab10")

plt.xlim(x0, x1)
plt.ylim(y0, y1)

m = Basemap(llcrnrlon=x0, llcrnrlat=y0, urcrnrlon=x1, urcrnrlat=y1, lat_0=cy, lon_0=cx, resolution='c',
            projection='mill')
# m = Basemap(lat_1=x0, lon_1=y0, lat_2=x1, lon_2=y1, lat_0=cx, lon_0=cy, resolution='c', projection='mill')

# Avoid border around map.
# m.drawmapboundary()

print("Reading shape file...")
m.readshapefile(shpfile, 'metro', linewidth=.15)
# plt.annotate(description, xy=(.58, 0.005), size=16, xycoords='axes fraction', color=fontcolor)

m.drawmapboundary()
m.drawparallels(np.arange(39, 42, .2), labels=[1, 1, 1, 1])
m.drawmeridians(np.arange(-75, -70, .2), labels=[1, 1, 1, 1])

print("Plotting...")
# plot nodes
nodes = [NodePosition(-74.00516, 40.74046),
         NodePosition(-73.99961, 40.73828),
         NodePosition(-73.99421, 40.73638),
         NodePosition(-73.98887, 40.73420),
         NodePosition(-73.98261, 40.73152),
         NodePosition(-73.97780, 40.72979)]

nodes_x = []
nodes_y = []
for node in nodes:
    nodes_x.append(node.x)
    nodes_y.append(node.y)

x, y = m(nodes_x, nodes_y)

for i in range(len(nodes)):
    m.scatter(x[i], y[i], marker='D', color=cmap_def(i))
# ax.scatter(nodes_x, nodes_y, s=2000, facecolors='none', edgecolors='red')

areas_x = []
areas_y = []

for i, node in enumerate(nodes):
    areas_x_node = []
    areas_y_node = []
    polycircle = polycircles.Polycircle(latitude=node.y, longitude=node.x, radius=1000, number_of_vertices=1000)
    for point in polycircle.to_lat_lon():
        areas_x.append(point[1])
        areas_x_node.append(point[1])
        areas_y.append(point[0])
        areas_y_node.append(point[0])

    x, y = m(areas_x_node, areas_y_node)
    m.scatter(x, y, s=1, facecolor=cmap_def(i))

plt.savefig(imgfile, bbox_inches='tight', pad_inches=.2)
plt.show()
