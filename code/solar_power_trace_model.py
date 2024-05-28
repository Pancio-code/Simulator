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

import math

import matplotlib.pyplot as plt
import numpy as np

from log import Log
from solar_power_model import SolarPowerModel


class SolarPowerTraceModel(SolarPowerModel):
    _MODULE = "SolarPowerModel"

    def __init__(self, raw_path=None, power_col=1,
                 scale_x=False, scale_y=False,
                 scale_to_max_x=1000.0, scale_to_min_x=0.0,
                 scale_to_max_y=10.0, scale_to_min_y=0.0,
                 cycles=1, shift=.0, parsed_x_limit=None, efficiency=1.0, to_watt_multiplier=1.0):

        self._raw_path = raw_path
        self._raw_data = None

        self._raw_data_x_max = 0.0
        self._raw_data_y_max = 0.0
        self._raw_data_x_max_forced = parsed_x_limit

        self._scale_x = scale_x
        self._scale_y = scale_y
        self._scale_to_max_y = scale_to_max_y
        self._scale_to_min_y = scale_to_min_y
        self._scale_to_max_x = scale_to_max_x
        self._scale_to_min_x = scale_to_min_x

        self._cycles = cycles
        self._shift = shift
        self._power_col = power_col

        self._efficiency = efficiency
        self._to_watt_multiplier = to_watt_multiplier

        self._parse()

    def _parse(self):
        self._raw_data = np.genfromtxt(self._raw_path, delimiter=",", dtype=float, skip_header=1)

        self._raw_data_x_max = self._raw_data_x_max_forced
        if self._raw_data_x_max_forced is None:
            self._raw_data_x_max = max(self._raw_data[:, 0])
        self._raw_data_x_min = min(self._raw_data[:, 0])

        self._raw_data_y_max = max(self._raw_data[:, self._power_col])
        self._raw_data_y_min = min(self._raw_data[:, self._power_col])

        Log.minfo(SolarPowerTraceModel._MODULE, f"parsed raw file: "
                                                f"power_col={self._power_col}, "
                                                f"x_max={self._raw_data_x_max}, "
                                                f"y_max={self._raw_data_y_max}, "
                                                f"x_min={self._raw_data_x_min}, "
                                                f"y_min={self._raw_data_y_min}, "
                                                f"scale_x_max={self._scale_to_max_x}, "
                                                f"scale_y_max={self._scale_to_max_y}, "
                                                f"scale_x_min={self._scale_to_min_x}, "
                                                f"scale_y_min={self._scale_to_min_y}, "
                                                f"efficiency={self._efficiency}")

        if math.isnan(self._raw_data_y_max) \
                or math.isnan(self._raw_data_y_min) \
                or math.isnan(self._raw_data_x_min) \
                or math.isnan(self._raw_data_x_max):
            Log.mfatal(SolarPowerTraceModel._MODULE, "_parse: parse data error")

    @staticmethod
    def _scale_value(x, from_min, from_max, to_min, to_max):
        return ((x - from_min) * ((to_max - to_min) / (from_max - from_min))) + to_min

    def get_watt_power_at(self, x):
        def normalize_x(x_value):
            if self._scale_x:
                normalized_x = SolarPowerTraceModel._scale_value(x_value,
                                                                 self._scale_to_min_x,
                                                                 self._scale_to_max_x,
                                                                 self._raw_data_x_min,
                                                                 self._raw_data_x_max)
                return (normalized_x * self._cycles) % self._raw_data_x_max
            return x_value

        def normalize_y(y_value):
            if self._scale_y:
                return self._efficiency * SolarPowerTraceModel._scale_value(y_value,
                                                                            self._raw_data_y_min,
                                                                            self._raw_data_y_max,
                                                                            self._scale_to_min_y,
                                                                            self._scale_to_max_y)
            return y_value

        y_value_raw = np.interp(
            normalize_x(x + self._shift),
            self._raw_data[:, 0],
            self._raw_data[:, self._power_col]
        )

        return normalize_y(y_value_raw) * self._to_watt_multiplier


def test():
    t = SolarPowerTraceModel("./solar_traces/data2.csv", power_col=3,
                             scale_x=True, scale_y=True,
                             scale_to_max_x=1000, scale_to_min_x=0,
                             scale_to_max_y=0.03, scale_to_min_y=0,
                             cycles=1, shift=0, parsed_x_limit=None)

    x = []
    y = []

    for i in range(1000):
        x.append(i)
        y.append(t.get_watt_power_at(i))

    plt.plot(x, y)
    plt.show()
    plt.close()


def test2():
    self_x1_min = 20
    self_x1_max = 80
    self_x2_min = 30
    self_x2_max = 40

    for x in range(20, 81):
        print(((x - self_x1_min) * ((self_x2_max - self_x2_min) / (self_x1_max - self_x1_min))) + self_x2_min)

# test()
