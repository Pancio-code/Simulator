#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import os
import shutil
import signal
import sys
import multiprocessing
from datetime import datetime
import datetime as dt

import simpy

from cloud import Cloud
from log import Log
from models import SolarPanelSpec
from node import Node
from service_data_storage import ServiceDataStorage
from service_discovery import ServiceDiscovery

"""
Run the simulation of deadline scheduling
"""

MODULE = "Main"

SIMULATION_TIME = 10000 #int(2 * 24 * 3600)
SIMULATION_TOTAL_TIME = SIMULATION_TIME
SOLAR_PANEL_ENABLED = False

SESSION_ID = datetime.now().strftime("%Y%m%d")
LEARNING_TYPE = Node.LearningType.D_SARSA
NO_LEARNING_POLICY = Node.NoLearningPolicy.RANDOM
ACTIONS_SPACE = Node.ActionsSpace.ONLY_WORKERS


ALPHA_INCREMENT = 0.05

# Define a function to run the simulation for a given alpha
def run_simulation(alpha):
    session_id = f'{SESSION_ID}_{alpha:.2f}'

    env = simpy.Environment()
    nodes = []
    
    tilt_list = [i for i in range(0, 72, 8)]
    azimuth_list = [i for i in range(0, 360, 40)]
    
    nodes_id_list = [1, 2, 3]
    
    tilt_list = [tilt_list[i] for i in nodes_id_list]
    azimuth_list = [azimuth_list[i] for i in nodes_id_list]
    
    panels_mapping = gen_spec_solar_panels(len(tilt_list), SIMULATION_TIME, "12-01-2020",
                                                            latitude_list=None,  # latitude_list,
                                                            longitude_list=None,  # longitude_list,
                                                            altitude_list=None,  # altitude_list,
                                                            tilt_list=tilt_list,
                                                            azimuth_list=azimuth_list,
                                                            efficiency=0.2,
                                                            panel_surface_m2=.4 * .4,
                                                            station_file="723170TYA.CSV")
    solar_panel_spec_by_node_id = {}
        
    if SOLAR_PANEL_ENABLED:
        for panel_spec in panels_mapping:
            solar_panel_spec_by_node_id[panel_spec.node_id] = panel_spec
            
    # create nodes
    cloud = Cloud(env, latency_roundtrip_ms=20)
    
    #1.8 1.7 1.4

    nodes.append(create_node(env, 0, 0, Node.NodeType.SCHEDULER, 1.0, 10, alpha))
    nodes.append(create_node(env, 1, 0, Node.NodeType.WORKER, 1.8, 9, alpha, solar_panel_spec_by_node_id.get(1)))
    nodes.append(create_node(env, 2, 0, Node.NodeType.WORKER, 1.7, 8, alpha , solar_panel_spec_by_node_id.get(2)))
    nodes.append(create_node(env, 3, 0, Node.NodeType.WORKER, 1.4, 7, alpha, solar_panel_spec_by_node_id.get(3)))

    
    # add them discovery service
    discovery = ServiceDiscovery(1, nodes, cloud)
    data_storage = ServiceDataStorage(nodes, session_id, LEARNING_TYPE, NO_LEARNING_POLICY, ACTIONS_SPACE)

    # init nodes services, and data
    for node in nodes:
        node.set_service_discovery(discovery)
        node.set_service_data_storage(data_storage)
    for node in nodes:
        node.init()
    cloud.set_service_discovery(discovery)

    Log.minfo(MODULE, f"Started simulation for alpha={alpha}")
    env.run(until=SIMULATION_TOTAL_TIME)
    Log.minfo(MODULE, f"Simulation ended: SESSION_ID={session_id}, LEARNING_TYPE={LEARNING_TYPE.name}, "
                      f"NO_LEARNING_POLICY={NO_LEARNING_POLICY.name}, ACTIONS_SPACE={ACTIONS_SPACE.name}, "
                      f"ALPHA={alpha}")

    data_storage.done_simulation()


def get_die_after(node_id):
    if node_id == 1:
        return 4000
    return 0


def get_die_simulation(node_id):
    if node_id == 1:
        return True
    return False


def create_node(env, node_id, belong_to_cluster_id, node_type, machine_speed,batt, alpha, node_solar_panel_spec=None):
    return Node(env,
                node_id,
                SESSION_ID,
                simulation_time=SIMULATION_TIME,
                skip_plots=True,
                node_belong_to_cluster=belong_to_cluster_id,
                node_type=node_type,
                # die_simulation=get_die_simulation(node_id),
                die_after_seconds=get_die_after(node_id),
                die_duration=4000,
                # rates
                machine_speed=machine_speed,
                rate_l=30.0,
                # solar panel
                solar_panel_enabled=SOLAR_PANEL_ENABLED,
                solar_panel_spec=node_solar_panel_spec,
                # traffic model
                # rate_l_model_path="./traffic/namex/namex-traffic-daily-20210420.csv",
                # rate_l_model_path=f"./traffic/fixed/fixed_{i}.csv",
                # rate_l_model_path=f"./traffic/city/data/traffic_node_{i}.csv",
                # rate_l_model_path="./traffic/fictious/fictious_1.csv",
                rate_l_model_path_shift=0,  # i * 1200,  # 0,
                rate_l_model_path_cycles=3,
                rate_l_model_path_parse_x_max=None,
                rate_l_model_path_steady=False,
                rate_l_model_path_steady_for=2000,
                rate_l_model_path_steady_every=2000,
                # net
                net_speed_client_scheduler_mbits=200,
                net_speed_scheduler_scheduler_mbits=300,
                net_speed_scheduler_worker_mbits=1000,
                net_speed_scheduler_cloud_mbits=1000,
                # job info
                job_periodic_types=3,
                job_periodic_payload_sizes_mbytes=(0.050, 0.050, 0.050),
                job_periodic_duration_std_devs=(0.0003, 0.0003, 0.0003),
                job_periodic_percentages=(.33, .33, .34),
                job_periodic_deadlines=(0.016, 0.033, 0.070),
                job_periodic_durations=(0.010, 0.020, 0.055),
                job_periodic_arrival_time_std_devs=(0.001, 0.002, 0.01),
                job_periodic_rates_fps=(60, 30, 15),
                job_periodic_desired_rates_fps=(60, 30, 15),
                job_periodic_desired_rates_fps_max=(60, 30, 15),
                job_periodic_desired_rates_fps_min=(50, 20, 10),
                job_exponential_types=1,
                job_exponential_payload_sizes_mbytes=[0.1],
                job_exponential_duration_std_devs=[0.01],
                job_exponential_arrival_time_std_devs=[0.01],
                job_exponential_percentages=[1],
                job_exponential_deadlines=[0.300],
                job_exponential_durations=[0.100],
                job_exponential_rates_fps=[10],
                job_exponential_desired_rates_fps=[1],
                job_exponential_desired_rates_fps_min=[0],
                job_exponential_desired_rates_fps_max=[10],
                # node info
                max_jobs_in_queue=5,
                distribution_arrivals=Node.DistributionArrivals.POISSON,
                delay_probing=0.003,
                # learning
                sarsa_alpha=0.01,
                sarsa_beta=0.01,
                state_type=Node.StateType.JOB_TYPE,
                learning_type=LEARNING_TYPE,
                no_learning_policy=NO_LEARNING_POLICY,
                actions_space=ACTIONS_SPACE,
                pwr2_binary_policy="001111",
                tiling_num_tilings=26,
                reward_alpha=alpha,
                # distributions
                distribution_network_probing_sigma=0.0001,
                distribution_network_forwarding_sigma=0.00002,
                episode_length=60,
                eps=0.90,
                eps_decay=0.9995,
                eps_dynamic=True,
                eps_min=0.05,
                logging_info=True,
                battery_total_capacity_wh=batt,
                battery_initial_capacity_wh=batt
                )
    
def gen_spec_solar_panels(n_nodes, simulation_time, start_date_str, latitude=41.80, longitude=12.36, altitude=5.0,
                        tilt_list = None, azimuth_list = None,
                        efficiency=0.20, panel_surface_m2=1.0,
                        station_file: str = "",
                        latitude_list = None,
                        longitude_list = None,
                        altitude_list = None):
    panels_specs = []
    for i in range(n_nodes):
        if latitude_list is not None and longitude_list is not None and altitude_list is not None:
            panels_specs.append(SolarPanelSpec(node_id=i, latitude=latitude_list[i], longitude=longitude_list[i],
                                                altitude=altitude_list[i], timezone=dt.timezone.utc,
                                                start_date_str=start_date_str,
                                                simulation_time_seconds=simulation_time, tilt=tilt_list[i],
                                                azimuth=azimuth_list[i], efficiency=efficiency,
                                                panel_surface_m2=panel_surface_m2,
                                                station_file=station_file))
        else:
            panels_specs.append(SolarPanelSpec(node_id=i, latitude=latitude, longitude=longitude, altitude=altitude,
                                                timezone=dt.timezone.utc, start_date_str=start_date_str,
                                                simulation_time_seconds=simulation_time, tilt=tilt_list[i],
                                                azimuth=azimuth_list[i], efficiency=efficiency,
                                                panel_surface_m2=panel_surface_m2,
                                                station_file=station_file))

    return panels_specs


if __name__ == "__main__":
    # Calculate the number of processes to launch based on CPU cores
    num_cores = multiprocessing.cpu_count()

    # Calculate the range of alphas to cover
    alpha_values = [i * ALPHA_INCREMENT for i in range(0, 21)]
    
    # Launch processes
    processes = []
    for alpha in alpha_values:
        process = multiprocessing.Process(target=run_simulation, args=(alpha,))
        processes.append(process)
        process.start()

        # Control the number of running processes
        if len(processes) >= num_cores:
            for p in processes:
                p.join()
            processes = []

    # Wait for remaining processes to finish
    for p in processes:
        p.join()

    print("All simulations completed.")