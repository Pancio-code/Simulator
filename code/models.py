#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import dataclasses
import datetime
from typing import List


@dataclasses.dataclass
class NodeSpec:
    id: int
    l: float  # lambda
    m: float  # mu
    neighbors: List[int]

    battery_capacity_wh: float  # in wh
    battery_fill_perc: float  # in %


@dataclasses.dataclass
class SolarTraceSpec:
    trace_id: int
    node_id: int

    scale_x: bool
    scale_x_min: float
    scale_x_max: float

    scale_y: bool
    scale_y_min: float
    scale_y_max: float

    efficiency: float
    cycles: int
    shift: float

    def get_obj(self):
        return {
            "trace_id": self.trace_id,
            "node_id": self.node_id,

            "scale_x": self.scale_x,
            "scale_x_min": self.scale_x_min,
            "scale_x_max": self.scale_x_max,

            "scale_y": self.scale_y,
            "scale_y_min": self.scale_y_min,
            "scale_y_max": self.scale_y_max,

            "efficiency": self.efficiency,
            "cycles": self.cycles,
        }


@dataclasses.dataclass
class SolarPanelSpec:
    node_id: int

    latitude: float
    longitude: float
    altitude: float

    timezone: datetime.timezone
    start_date_str: str

    simulation_time_seconds: int

    tilt: int
    azimuth: int
    efficiency: float
    panel_surface_m2: float

    station_file: str

    def get_obj(self):
        return {
            "node_id": self.node_id,

            "latitude": self.latitude,
            "longitude": self.longitude,
            "altitude": self.altitude,

            "timezone": self.timezone,
            "start_date_str": self.start_date_str,

            "simulation_time_seconds": self.simulation_time_seconds,

            "tilt": self.tilt,
            "azimuth": self.azimuth,
            "efficiency": self.efficiency,
            "panel_surface_m2": self.panel_surface_m2,

            "station_file": self.station_file
        }
