#  Copyright (c) 2024. Gabriele Proietti Mattia <pm.gabriele@gmail.com>
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

import datetime
import pathlib

import pandas as pd 
import pvlib
from pvlib import irradiance, iotools
from pvlib import location

from log import Log
from models import SolarPanelSpec
from solar_power_trace_model import SolarPowerModel


class SolarPowerPanelModel(SolarPowerModel):
    _MODULE = "SolarPanelPvLib"

    def __init__(self, node_id=0, latitude=39.55, longitude=105.221, altitude=0.0, timezone=datetime.timezone.utc,
                 start_date_str='06-20-2020', simulation_time_secs=1000, tilt=25, surface_azimuth=180,
                 efficiency=0.2, panel_surface_m2=1.0, spec: SolarPanelSpec or None = None, station_file=""):
        if spec is not None:
            node_id = spec.node_id
            latitude = spec.latitude
            longitude = spec.longitude
            altitude = spec.altitude
            timezone = spec.timezone
            start_date_str = spec.start_date_str
            simulation_time_secs = spec.simulation_time_seconds
            tilt = spec.tilt
            surface_azimuth = spec.azimuth
            efficiency = efficiency
            panel_surface_m2 = spec.panel_surface_m2
            station_file = spec.station_file

        simulation_time_secs += simulation_time_secs * 0.1  # add 10% to the simulation time
        print(f"simulation_time_secs={simulation_time_secs}")

        self._start_date = (
            datetime.datetime.strptime(start_date_str, '%m-%d-%Y')
            .replace(tzinfo=timezone, hour=0, minute=0, second=0, microsecond=0))
        self._end_date = self._start_date + datetime.timedelta(seconds=simulation_time_secs)
        self._site = None
        self._times = None
        self._weather = None
        

        if station_file is not None and len(station_file) > 0:  # '723170TYA.CSV'
            Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: using station_file={station_file} spec={spec}")

            coerce_year = 1990
            DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
            tmy, metadata = iotools.read_tmy3(DATA_DIR / station_file, coerce_year=coerce_year, map_variables=True)
            # tmy.index = tmy.index - pd.Timedelta(hours=1)
            self._weather = pd.DataFrame({
                'ghi': tmy['ghi'], 'dhi': tmy['dhi'], 'dni': tmy['dni'],
                'temp_air': tmy['temp_air'], 'wind_speed': tmy['wind_speed'],
            })
            self._site = location.Location.from_tmy(metadata)
            timezone = datetime.timezone(datetime.timedelta(hours=self._site.tz))

            # change the timezone of the input dates
            self._start_date = (datetime.datetime.strptime(start_date_str, '%m-%d-%Y')
                                .replace(tzinfo=timezone, year=coerce_year))
            self._end_date = self._start_date + datetime.timedelta(seconds=simulation_time_secs)  # round to ceil hour
            # Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: self._site.tz={self._site.tz} site={self._site}")
            # Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: self._start_date={self._start_date} self._end_date={self._end_date}")

            # filter the data to the given range
            # filter to ceil hour (since station data is hour by hour)
            self._weather = self._weather[
                self._weather.index.to_series().between(self._start_date, self._end_date + datetime.timedelta(hours=1))]
            # interpolate
            self._weather = self._weather.resample("s").interpolate()
            # filter to true time
            self._weather = self._weather[self._weather.index.to_series().between(self._start_date, self._end_date)]
            # save the index
            self._times = self._weather.index
            # Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: self._weather.index={self._weather.index}")
        else:
            # Create location object to store lat, lon, timezone
            self._site = location.Location(latitude=latitude, longitude=longitude, altitude=altitude, tz=timezone)
            # Generate clearsky data using the Ineichen model, which is the default
            # The get_clearsky method returns a dataframe with values for GHI, DNI,
            # and DHI
            self._times = pd.date_range(start=self._start_date, end=self._end_date, freq='1s', tz=self._site.tz)
            clearsky = self._site.get_clearsky(self._times)
            self._weather = clearsky

        # Get solar azimuth and zenith to pass to the transposition function
        solar_position = self._site.get_solarposition(times=self._times)
        # Use the get_total_irradiance function to transpose the GHI to POA
        poa_irradiance = irradiance.get_total_irradiance(
            surface_tilt=tilt,
            surface_azimuth=surface_azimuth,
            dni=self._weather['dni'],
            ghi=self._weather['ghi'],
            dhi=self._weather['dhi'],
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'])

        self._ghi = self._weather['ghi']
        self._poa = poa_irradiance['poa_global']

        # data = pd.DataFrame({'GHI': self._ghi, 'POA': self._poa})
        # Log.mdebug(SolarPowerPanelModel._MODULE, f"data=\n{data}")

        # resample to seconds and interpolate values
        # self._poa = self._poa.resample("s").interpolate()
        # Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: poa_irradiance={self._poa}")
        # Log.mdebug(SolarPowerPanelModel._MODULE, f"__init__: ghi={self._ghi}")

        self._efficiency = efficiency
        self._panel_surface_m2 = panel_surface_m2
        
        

        Log.minfo(SolarPowerPanelModel._MODULE, f"__init__: "
                                                f"node_id={node_id} "
                                                f"start_date={self._start_date} "
                                                f"end_date={self._end_date} "
                                                f"simulation_time={simulation_time_secs}")

    def get_watt_power_at(self, simulation_seconds):
        simulation_date = self._start_date + datetime.timedelta(seconds=simulation_seconds)
        instant_power = self._poa[simulation_date] * self._efficiency * self._panel_surface_m2
        return instant_power


if __name__ == '__main__':
    lat, lon = 41.81, 12.36  # Roma, Ponte Galeria
    panel = SolarPowerPanelModel(latitude=lat, longitude=lon, altitude=5.0, simulation_time_secs=24 * 60 * 60 + 1323,
                                 start_date_str="06-20-1990", panel_surface_m2=.4 * .4, station_file="723170TYA.CSV")
    print(f"panel.get_power_at({0})={panel.get_watt_power_at(0)}")
    print(f"panel.get_power_at({8 * 60 * 60})={panel.get_watt_power_at(9 * 60 * 60)}")
    print(f"panel.get_power_at({9 * 60 * 60})={panel.get_watt_power_at(9 * 60 * 60)}")
    print(f"panel.get_power_at({10 * 60 * 60})={panel.get_watt_power_at(10 * 60 * 60)}")
    print(f"panel.get_power_at({11 * 60 * 60})={panel.get_watt_power_at(11 * 60 * 60)}")
    print(f"panel.get_power_at({12 * 60 * 60})={panel.get_watt_power_at(12 * 60 * 60)}")
    print(f"panel.get_power_at({13 * 60 * 60})={panel.get_watt_power_at(13 * 60 * 60)}")
    print(f"panel.get_power_at({24 * 60 * 60})={panel.get_watt_power_at(13 * 60 * 60)}")
    print(f"panel.get_power_at({24 * 60 * 60 + 1323})={panel.get_watt_power_at(24 * 60 * 60 + 1323)}")
    # print(f"panel.get_power_at({19999})={panel.get_watt_power_at(19999)}")
    # print(f"panel.get_power_at({20000})={panel.get_watt_power_at(20000)}")
