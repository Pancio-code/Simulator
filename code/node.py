#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import os
import pickle
import random
from statistics import variance
import sys
import time
import traceback
from enum import Enum
from typing import List

import numpy as np
import simpy

from function_approximation import DSPSarsaTiling
from job import Job
from log import Log
from traffic_model import TrafficModel
from solar_power_panel_model import SolarPowerPanelModel
from solar_power_trace_model import SolarPowerTraceModel

"""
Implementation of a node in a fog environment
"""

DEBUG = False
MODULE = "Node"


class DiscoveryType:
    NO_COOPERATION = 0
    BEST_DEADLINE_ORACLE = 1
    BEST_DEADLINE_PWR_1 = 2
    BEST_DEADLINE_PWR_2 = 3
    PWR_1 = 4
    LEAST_LOADED = 5


# noinspection DuplicatedCode
class Node:
    MODULE = "Node"

    BASE_DIR_LOG = "_log"
    BASE_DIR_PLOT = "_plot"
    BASE_DIR_DNN = "_dnn"
    BASE_DIR_MODELS = "_models"
    BASE_DIR_TB = "_scalars"

    LOG_KEY_EPISODE = "episode"
    LOG_KEY_EPS = "eps"
    LOG_KEY_SCORE = "score"
    LOG_KEY_TOTAL_JOBS = "total_jobs"
    LOG_KEY_LOSS = "loss"
    LOG_KEY_MSE = "mse"
    LOG_KEY_MAE = "mae"

    LOG_KEYS = [LOG_KEY_EPISODE, LOG_KEY_EPS, LOG_KEY_SCORE, LOG_KEY_TOTAL_JOBS, LOG_KEY_LOSS, LOG_KEY_MSE, LOG_KEY_MAE]

    LOG_POS_EPISODE = 0
    LOG_POS_EPS = 1
    LOG_POS_SCORE = 2
    LOG_POS_TOTAL_JOBS = 3
    LOG_POS_LOSS = 4
    LOG_POS_MSE = 5
    LOG_POS_MAE = 6

    LOG_KEYS_POS = [LOG_POS_EPISODE, LOG_POS_EPS, LOG_POS_SCORE, LOG_POS_TOTAL_JOBS, LOG_POS_LOSS, LOG_POS_MSE,
                    LOG_POS_MAE]

    class ActionsType(Enum):
        """Types of actions that the agent can perform"""
        ALL = 0  # all actions, i.e. schedule to all nodes
        AVAILABLE = 1  # only actions that are valid according to _get_actions
        AVAILABLE_AWARE = 2  # only actions that are valid according to _get_actions and the deadline

    class ActionsSpace(Enum):
        WORKERS_OR_CLOUD = 0  # total actions: number of nodes + 1
        ONLY_WORKERS = 1  # # total actions: number of nodes
        OTHER_CLUSTERS = 2  # total actions: number of nodes + 1 + n_clusters

    class LearningType(Enum):
        NO_LEARNING = 0
        Q_DNN = 1
        Q_TABLE = 2
        D_SARSA = 3

    class NoLearningPolicy(Enum):
        RANDOM = 0
        ROUND_ROBIN = 1
        LEAST_LOADED = 2
        LEAST_LOADED_AWARE = 3
        LEAST_LOADED_NOT_AWARE = 4
        LEAST_LOADED_AWARE_CLOUD = 5
        MAXIMUM_LIFESPANE = 6

    class DistributionArrivals(Enum):
        POISSON = 0
        DETERMINISTIC = 1
        GAUSSIAN = 2

    class DistributionNetworkLatency(Enum):
        DETERMINISTIC = 0
        GAUSSIAN = 1

    class StateType(Enum):
        ONLY_NUMBER = 0  # state is only the number of queued jobs
        JOB_TYPE = 1  # state is the number of queued job for each type

    class NodeType(Enum):
        SCHEDULER = 0
        WORKER = 1

    class JobArrivalType(Enum):
        EXPONENTIAL = 0
        PERIODIC = 1

    # noinspection PyUnresolvedReferences,PyDefaultArgument
    def __init__(self, env, uid, session_uid,
                 simulation_time=50000,
                 discovery=None,
                 data_storage=None,
                 # machine
                 max_jobs_in_queue=10,
                 machine_speed=1.0,  # 1.0 means that job duration is equal to the one assigned
                 node_type=NodeType.WORKER,
                 node_belong_to_cluster=None,
                 die_simulation=False,
                 die_after_seconds=3000,
                 die_duration=500,
                 round_time_s=100,
                 # jobs
                 job_duration_type=Job.DurationType.GAUSSIAN,
                 job_periodic_types=3,
                 job_periodic_payload_sizes_mbytes=(0.1, 0.1),
                 job_periodic_duration_std_devs=(0.0013, 0.0013),
                 job_periodic_percentages=(.33, .33, .34),
                 job_periodic_deadlines=(0.016, 0.033, 0.70),
                 job_periodic_durations=(0.005, 0.005, 0.005),
                 job_periodic_arrival_time_std_devs=(0.005, 0.005, 0.005),
                 job_periodic_rates_fps=(60, 30, 15),
                 job_periodic_desired_rates_fps=(60, 30, 15),
                 job_periodic_desired_rates_fps_max=(60, 30, 15),
                 job_periodic_desired_rates_fps_min=(55, 25, 10),
                 job_exponential_types=1,
                 job_exponential_payload_sizes_mbytes=[0.1],
                 job_exponential_duration_std_devs=[0.0013],
                 job_exponential_arrival_time_std_devs=[0.0013],
                 job_exponential_percentages=[1],
                 job_exponential_deadlines=[10],
                 job_exponential_durations=[0.100],
                 job_exponential_rates_fps=[2],
                 job_exponential_desired_rates_fps=[1],
                 job_exponential_desired_rates_fps_min=[0],
                 job_exponential_desired_rates_fps_max=[10],
                 # network
                 net_speed_client_scheduler_mbits=200,
                 net_speed_scheduler_scheduler_mbits=300,
                 net_speed_scheduler_worker_mbits=1000,
                 net_speed_scheduler_cloud_mbits=1000,
                 delay_probing=0.004,
                 # traffic
                 rate_l=1.0,
                 rate_l_model_path=None,
                 rate_l_model_path_cycles=1,
                 rate_l_model_path_shift=0,
                 rate_l_model_path_parse_x_max=None,
                 rate_l_model_path_steady=False,
                 rate_l_model_path_steady_for=2000,
                 rate_l_model_path_steady_every=2000,
                 # distributions
                 distribution_arrivals=DistributionArrivals.POISSON,
                 distribution_network_latency=DistributionNetworkLatency.GAUSSIAN,
                 distribution_network_probing_sigma=0.0002,
                 distribution_network_forwarding_sigma=0.00001,
                 distribution_network_latency_cloud_sigma=0.00001,
                 # energy
                 battery_total_capacity_wh=10,
                 battery_initial_capacity_wh=10,
                 power_max_cpu_w=3.0,
                 power_max_transmission_w=0.2,
                 power_idle_w=2.0,
                 energy_to_idle_thresholds_enabled=False,
                 energy_to_idle_high_threshold=2,
                 energy_to_idle_low_threshold=1,
                  # solar
                 solar_trace_enabled=False,
                 solar_trace_csv="",
                 solar_trace_spec=None,
                 solar_trace_csv_col=1,
                 solar_trace_to_watt_multiplier=1000,
                 solar_trace_efficiency=1.0,
                 # solar panel
                 solar_panel_enabled=False,
                 solar_panel_spec=None,
                 # actions
                 actions_space=ActionsSpace.WORKERS_OR_CLOUD,
                 actions_type=ActionsType.ALL,
                 # learning
                 sarsa_alpha=0.01,
                 sarsa_beta=0.01,
                 state_type=StateType.JOB_TYPE,
                 episode_length=None,
                 batch_size=0,
                 use_model_from_session_name=None,
                 eps=0.1, eps_min=0.05, eps_decay=0.999,
                 learning_type=LearningType.NO_LEARNING,
                 no_learning_policy=NoLearningPolicy.RANDOM,
                 eps_dynamic=False,
                 tiling_num_tilings=16,
                 reward_alpha=1.0,
                 # other parameters
                 skip_plots=False,
                 pwr2_threshold=3,
                 pwr2_binary_policy=None,  # express like '11111',
                 # utils
                 logging_info=True
                 ):

        #
        # Fixed variables
        #

        self._env = env  # type: 'simpy.Environment'

        # node parameters
        self._simulation_time = simulation_time
        """Total time of the simulation"""
        self._uid = random.randint(0, 100000) if uid is None else uid
        """Node id"""
        self._session_uid = session_uid
        """Session id"""

        if self._session_uid is None:
            Log.merr(f"{MODULE}#{self._uid}", f"Node#{id}: session_uid cannot be none")
            exit(1)

        # machine
        self._max_jobs_in_queue = max_jobs_in_queue  # = to 'k'
        """Maximum number of jobs in the queue"""
        self._machine_speed = machine_speed
        """Multiplier of the job duration before applying the distribution function"""
        self._node_type = node_type
        self._node_belong_to_cluster = node_belong_to_cluster
        if self._node_belong_to_cluster is None:
            Log.mfatal(f"Cluster cannot be None for node {self.get_uid()}")
            sys.exit(1)

        # traffic
        self._rate_l = rate_l
        """Job arrival rate"""

        # traffic model
        self._traffic_model = TrafficModel(
            raw_path=rate_l_model_path,
            max_x=simulation_time,
            cycles=rate_l_model_path_cycles,
            shift=rate_l_model_path_shift,
            parsed_x_limit=rate_l_model_path_parse_x_max,
            steady=rate_l_model_path_steady,
            steady_for=rate_l_model_path_steady_for,
            steady_every=rate_l_model_path_steady_every,
        ) if rate_l_model_path is not None else None

        # network
        self._delay_probing = delay_probing  # s
        self._net_speed_client_scheduler_mbits = net_speed_client_scheduler_mbits
        self._net_speed_scheduler_scheduler_mbits = net_speed_scheduler_scheduler_mbits
        self._net_speed_scheduler_worker_mbits = net_speed_scheduler_worker_mbits
        self._net_speed_scheduler_cloud_mbits = net_speed_scheduler_cloud_mbits

        # jobs
        self._job_duration_type = job_duration_type
        self._job_periodic_types = job_periodic_types
        self._job_periodic_payload_sizes_mbytes = job_periodic_payload_sizes_mbytes
        self._job_periodic_duration_std_devs = job_periodic_duration_std_devs
        self._job_periodic_percentages = job_periodic_percentages
        self._job_periodic_deadlines = job_periodic_deadlines
        self._job_periodic_durations = job_periodic_durations
        self._job_periodic_rates_fps = job_periodic_rates_fps
        self._job_periodic_arrival_time_std_devs = job_periodic_arrival_time_std_devs
        self._job_periodic_desired_rates_fps = job_periodic_desired_rates_fps
        self._job_periodic_desired_rates_fps_max = job_periodic_desired_rates_fps_max
        self._job_periodic_desired_rates_fps_min = job_periodic_desired_rates_fps_min

        self._job_exponential_types = job_exponential_types
        self._job_exponential_payload_sizes_mbytes = job_exponential_payload_sizes_mbytes
        self._job_exponential_duration_std_devs = job_exponential_duration_std_devs
        self._job_exponential_percentages = job_exponential_percentages
        self._job_exponential_deadlines = job_exponential_deadlines
        self._job_exponential_durations = job_exponential_durations
        self._job_exponential_rates_fps = job_exponential_rates_fps
        self._job_exponential_arrival_time_std_devs = job_exponential_arrival_time_std_devs
        self._job_exponential_desired_rates_fps = job_exponential_desired_rates_fps
        self._job_exponential_desired_rates_fps_max = job_exponential_desired_rates_fps_max
        self._job_exponential_desired_rates_fps_min = job_exponential_desired_rates_fps_min

        if sum(self._job_periodic_percentages) != 1.0 or sum(self._job_exponential_percentages) != 1.0:
            raise ValueError(f"Job percentages is not summing to 1: {job_percentages}")

        # actions
        self._state_type = state_type
        self._actions_type = actions_type
        self._actions_space = actions_space
        """The type of actions among which the agent can choose"""
        # learning
        self._epsilon = eps  # exploration rate
        """Exploration rate"""
        self._epsilon_min = eps_min
        self._epsilon_decay = eps_decay
        self._eps_dynamic = eps_dynamic
        """If eps should be updated automatically"""
        self._session_learning_type = learning_type
        self._session_no_learning_policy = no_learning_policy
        self._use_model_from_session_name = use_model_from_session_name

        # distribution
        self._distribution_network_latency = distribution_network_latency
        self._distribution_network_probing_sigma = distribution_network_probing_sigma
        self._distribution_network_forwarding_sigma = distribution_network_forwarding_sigma
        self._distribution_network_latency_cloud_sigma = distribution_network_latency_cloud_sigma
        self._distribution_arrivals = distribution_arrivals
        """The statistical distribution for arrival jobs"""

        # other
        self._pwr2_threshold = pwr2_threshold
        self._pwr2_binary_policy = pwr2_binary_policy
        self._skip_plots = skip_plots

        #
        # Runtime variables
        #

        # services
        self._service_discovery = discovery  # type: 'DiscoveryService'
        """Discovery service"""
        self._service_data_storage = data_storage  # type: 'DataStorage'
        """Data storage service"""

        # containers
        self._queued_jobs = simpy.resources.container.Container(self._env, capacity=self._max_jobs_in_queue, init=0)
        """Queued jobs container"""
        self._queued_probe_jobs = simpy.resources.container.Container(self._env, init=0)
        """Queued probing jobs container"""
        self._queued_transmission_jobs = simpy.resources.container.Container(self._env, init=0)
        """Queued transmission jobs container"""

        # die simulator
        self._die_simulation = die_simulation
        self._die_after_seconds = die_after_seconds
        self._die_duration = die_duration
        self._die_is_died = False

        self._process_generators = []
        """Process of job generator"""

        # processes
        if self._node_type == Node.NodeType.WORKER:
            self._process_logger_battery = self._env.process(self._process_logger_batt())  # type: simpy.Process
            self._process_worker = self._env.process(self._process_job_executor())  # type: simpy.Process
            self._process_energy_manager = self._env.process(self._process_energy())  # type: simpy.Process
            """Process of job processor"""
            if self._die_simulation:
                self._process_die_simulator = self._env.process(self._process_die_simulator())  # type: simpy.Process
        if self._node_type == Node.NodeType.SCHEDULER:
            Log.minfo(self._module(), f"__init__: periodic_types={self._job_periodic_types}")
            for i in range(self._job_periodic_types):
                Log.minfo(self._module(), f"__init__: periodic_types={self._job_periodic_types}, i={i}")
                p = self._env.process(self._process_jobs_generator(Node.JobArrivalType.PERIODIC, i))  # type: simpy.Process
                self._process_generators.append(p)

            Log.minfo(self._module(), f"__init__: exp_types={self._job_exponential_types}")
            for i in range(self._job_exponential_types):
                p = self._env.process(self._process_jobs_generator(Node.JobArrivalType.EXPONENTIAL, i))  # type: simpy.Process
                self._process_generators.append(p)

        # """Process that simulates probing requests"""
        self._process_transmission = self._env.process(self._process_job_transmission())  # type: simpy.Process
        """Process that simulates probing requests"""
        self._process_logger = self._env.process(self._process_logger_impl())  # type: simpy.Process
        """Process that simulates probing requests"""
        # self._process_shutdown = self._env.process(self._process_node_shutdown())  # type: simpy.Process
        """Process of job generator shutdown"""
        self._jobs_list = []
        """List of jobs in the queue"""
        self._jobs_probing_list = []
        """List of jobs enqueue for probing"""
        self._jobs_transmission_list = []
        """List of jobs enqueue for transmission"""
        self._currently_executing_job = None  # type: 'Job' or None
        """The current job that is executing"""
        self._loads_cluster = []  # this is [ [#jobstype1, #jobstype2], [..., ...] ]
        """The load of every worker node given the job that we assigned to them"""
        self._loads_our_history = [0]

        # counters and lists
        self._total_jobs = 0
        """Total number of dispatched jobs"""
        self._total_processed_job = 0
        """Total number of jobs memorized"""
        self._last_episode_end_at = 0
        """The number of job at which the last episode end"""
        self._scheduled_jobs = []  # type: List[Job]
        """List of episode jobs"""
        self._current_episode_number = 0
        """Number of total episodes"""
        self._last_logged_q_value_time = -1

        # logging
        self._data_log = {}  # episode, eps, score, total_jobs
        for key in self.LOG_KEYS:
            self._data_log[key] = []

        # to init later
        """Total number of nodes"""
        self._episode_length = episode_length  # to be init -- episode_length if episode_length is not None else self._n_nodes * 3
        """Total number of jobs for declaring the end of an episode"""
        self._batch_size = batch_size  # to be init -- batch_size if batch_size is not None else self._episode_length * 3
        """Total number of jobs for creating a batch when replay"""

        self._action_size = 0  # to be init -- self._n_nodes + 1
        """Size of the action space, all nodes + reject job (last action)"""
        self._states_number = 0  # to be init -- pow(self._n_nodes, self._params.k + 1)
        """Size of the state space"""
        self._all_possible_states = None

        # scheduling utils
        self._round_robin_counter = 0

        # learning
        self._tiling_num_tilings = tiling_num_tilings
        self._sarsa_alpha = sarsa_alpha
        self._sarsa_beta = sarsa_beta
        self._reward_alpha = reward_alpha

        # utils
        self._logging_info = logging_info

        # energy
        self._energy_to_idle_thresholds_enabled = energy_to_idle_thresholds_enabled
        """Check if to enable the thresholds for task execution on/off"""
        self._energy_to_idle_high_threshold = energy_to_idle_high_threshold
        """Energy in the battery to resume the inference"""
        self._energy_to_idle_low_threshold = energy_to_idle_low_threshold
        """Energy in the battery to stop the inference and go idle"""
        if self._energy_to_idle_low_threshold == self._energy_to_idle_high_threshold:
            Log.mwarn(self._module(), "__init__: energy_to_idle thresholds are equal")

        self._battery_total_capacity_wh = battery_total_capacity_wh
        self._battery_initial_capacity_wh = battery_initial_capacity_wh
        self._battery_current_capacity_wh = battery_initial_capacity_wh
        self._battery_last_round_capacity_wh = battery_initial_capacity_wh
        self._batteries_cluster = []

        self._power_max_cpu_w = power_max_cpu_w
        self._power_max_transmission_w = power_max_transmission_w
        self._power_idle_w = power_idle_w
        
        # solar energy
        self._solar_trace_enabled = solar_trace_enabled
        self._solar_trace_csv = solar_trace_csv
        self._solar_trace_spec = solar_trace_spec  # type: SolarTraceSpec
        self._solar_trace_csv_col = solar_trace_csv_col
        self._solar_trace_efficiency = solar_trace_efficiency
        self._solar_trace_to_watt_multiplier = solar_trace_to_watt_multiplier

        self._solar_power_model: SolarPowerModel or None = None
        if self._solar_trace_enabled:
            # create the panel model from the trace
            self._solar_power_model = SolarPowerTraceModel(
                raw_path=self._solar_trace_csv,
                power_col=self._solar_trace_csv_col,
                scale_x=self._solar_trace_spec.scale_x, scale_y=self._solar_trace_spec.scale_y,
                scale_to_max_x=self._solar_trace_spec.scale_x_max, scale_to_min_x=self._solar_trace_spec.scale_x_min,
                scale_to_max_y=self._solar_trace_spec.scale_y_max, scale_to_min_y=self._solar_trace_spec.scale_y_min,
                cycles=self._solar_trace_spec.cycles, shift=self._solar_trace_spec.shift, parsed_x_limit=None,
                efficiency=self._solar_trace_efficiency, to_watt_multiplier=self._solar_trace_to_watt_multiplier
            )

        # solar panel model
        self._solar_panel_enabled: bool = solar_panel_enabled
        """Check if a solar panel model is used"""
        self._solar_panel_spec = solar_panel_spec
        """Solar panel specifications"""
        # create the panel model
        if self._solar_panel_enabled:
            self._solar_power_model = SolarPowerPanelModel(spec=self._solar_panel_spec)

        self._round_step = 100
        self._metric_round_energy_list = []
        self._metric_round_energy_consumed = 0.0
        self._metric_round_energy_net = 0.0
        self._metric_round_energy_gained = 0.0
        self._time_energy_execution = 0.0
        self._time_energy_transmission = 0.0
        self._time_round_total = 0.0
        """Total time employed in task execution+queue from the end to the last round"""
        self._time_round_execution = 0.0
        """Total time employed in task execution from the end to the last round"""
        self._time_round_transmission = 0.0
        self._is_idle = False
        """Is this node forcefully idle?"""

        self._round_n = 0
        """Current round"""
        self._round_last_timestamp = 0
        """Timestamp of the last round"""
        self._round_time_s = round_time_s

        # runtime
        self._job_latest_processed = [None for _ in range(self._job_periodic_types + self._job_exponential_types)]  # type: List[Job or None]

        # checks
        if self._solar_panel_enabled and self._solar_trace_enabled:
            Log.mfatal(self._module(), "__init__: you enabled both panels and traces")

    #
    # Init
    #

    def init(self):
        """Init the node params after services have been installed"""
        workers = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        workers_n = len(workers)

        self._loads_cluster = [[0 for i in range(self._job_periodic_types + self._job_exponential_types)] for _ in
                               range(workers_n)]
        self._batteries_cluster = [worker.get_battery_residual_wh() for worker in workers]

        if self._actions_space is Node.ActionsSpace.ONLY_WORKERS:
            self._action_size = workers_n  # reject
        elif self._actions_space is Node.ActionsSpace.WORKERS_OR_CLOUD:
            self._action_size = workers_n + 1  # reject + cloud
        elif self._actions_space is Node.ActionsSpace.OTHER_CLUSTERS:
            self._action_size = workers_n + 1 + self._service_discovery.get_clusters_count() - 1 + 1  # reject, cloud, clusters
        else:
            Log.merr(f"{MODULE}#{self._uid}", "self._actions_space is not valid")
            raise RuntimeError("ActionsSpace not valid")

        self._episode_length = self._episode_length if self._episode_length > 0 else 20

        # init learning strategy and parameters
        if self._session_learning_type is not Node.LearningType.NO_LEARNING:
            self._batch_size = self._batch_size if self._batch_size > 0 else self._episode_length * 3
            self._states_number = self._max_jobs_in_queue * 2  # self._action_size  # self._max_jobs_in_queue  the current node state is the state

            if self._logging_info:
                Log.minfo(self._module(),
                          f"_init: episode_length={self._episode_length}, self._batch_size={self._batch_size}, "
                          f"self._action_size={self._action_size}, self._states_number={self._states_number}")

            # init learning strategy
            if self._session_learning_type == Node.LearningType.Q_DNN:
                # self._init_q_dnn()
                pass
            elif self._session_learning_type == Node.LearningType.Q_TABLE:
                # self._init_q_table()
                pass
            elif self._session_learning_type == Node.LearningType.D_SARSA:
                self._init_d_sarsa()

        # self._all_possible_states = self._get_all_possible_states_str()

        self._init_dirs()

        if self._logging_info:
            Log.minfo(f"{MODULE}#{self._uid}", f"Node#{self._uid} completed init for session#{self._session_uid}")

    def _init_dirs(self):
        if self._session_learning_type == Node.LearningType.NO_LEARNING:
            self._DIR_LOG = f"{Node.BASE_DIR_LOG}/no-learning/{self._session_no_learning_policy.name}/{self._session_uid}/Node#{self._uid}"
            self._DIR_PLOT = f"{Node.BASE_DIR_PLOT}/no-learning/{self._session_no_learning_policy.name}/{self._session_uid}/Node#{self._uid}"
        else:
            self._DIR_MODELS = f"{Node.BASE_DIR_MODELS}/{self._session_learning_type}/{self._session_uid}/Node#{self._uid}"
            self._DIR_TB_LOG = f"{Node.BASE_DIR_TB}/{self._session_learning_type}/{self._session_uid}/Node#{self._uid}"
            self._DIR_LOG = f"{Node.BASE_DIR_LOG}/learning/{self._session_learning_type.name}/{self._actions_space.name}/{self._session_uid}/Node#{self._uid}"
            self._DIR_PLOT = f"{Node.BASE_DIR_PLOT}/learning/{self._session_learning_type.name}/{self._actions_space.name}/{self._session_uid}/Node#{self._uid}"
            os.makedirs(self._DIR_MODELS, exist_ok=True)
            if self._session_learning_type == Node.LearningType.Q_DNN:
                pass

        os.makedirs(self._DIR_LOG, exist_ok=True)
        os.makedirs(self._DIR_PLOT, exist_ok=True)

    #
    # Processes
    #
    def _process_logger_batt(self):
        """Process used for logging values at specific times"""
        workers = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        while True:
            try:
                batteries = []
                for worker in workers:
                    batteries.append(worker.get_battery_residual_wh())
                    
                self._metric_round_energy_consumed = 0.0
                self._metric_round_energy_gained = 0.0
                self._metric_round_energy_net = 0.0
                    
                self._log_battery(batteries)
                yield self._env.timeout(self._round_time_s)

            except Exception as e:
                traceback.print_exc()
                Log.merr(f"{MODULE}#{self._uid}", f"Node#{self.get_uid()} _process_logger interrupted, e={e}")
                exit(1)

    def _process_job_executor(self):
        """Process which executes jobs for a node"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"job executor started")

        while True:
            try:
                # wait for a job
                yield self._queued_jobs.get(1)
                job = self._jobs_list.pop(0)
                self._currently_executing_job = job

                # compute the actual job duration according to parameters
                actual_job_duration = Job.compute_duration(self._currently_executing_job, self._job_duration_type,
                                                           self._machine_speed)

                # execute it
                yield self._env.timeout(actual_job_duration)

                # update exec time
                self._currently_executing_job.a_executed(actual_job_duration)
                # if job.get_type() == 1:
                # print(f"type={job.get_type()} job_duration={actual_job_duration}")

                self._currently_executing_job = None

                self._time_round_execution += actual_job_duration
                self._time_energy_execution += actual_job_duration

                # read to the transmission queue
                self._job_transmit(job)

                if DEBUG:
                    Log.mdebug(f"{MODULE}#{self._uid}",
                               f"_process_job_executor: executed job={job}, next_action={job.get_transmission_next_action()}, level={self._queued_jobs.level}")

            except Exception as e:
                # if DEBUG:
                if self._currently_executing_job is not None:
                    Log.merr(f"{MODULE}#{self._uid}",
                             f"Node#{self.get_uid()} job executor interrupted, currently executing job={self._currently_executing_job} e={e}")
                else:
                    Log.merr(f"{MODULE}#{self._uid}", f"Node#{self.get_uid()} job executor interrupted, e={e}")

    def _process_job_probing(self):
        """Process which simulates a probing"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"job executor started")

        def actual_time(delay):
            if self._distribution_network_latency == Node.DistributionNetworkLatency.GAUSSIAN:
                return random.gauss(delay, self._distribution_network_probing_sigma)
            return delay

        while True:
            try:
                # wait for a job
                yield self._queued_probe_jobs.get(1)
                job = self._jobs_probing_list.pop(0)

                # send the probing
                yield self._env.timeout(actual_time(self._delay_probing / 2))

                # pick the state of a random node
                # probe 1 random node
                random_node = self._service_discovery.get_random_node(current_node_id=self.get_uid())  # type: Node
                random_node_load = random_node.get_current_load()

                # return
                yield self._env.timeout(actual_time(self._delay_probing / 2))

                # check the load
                if random_node_load < job.get_state_snapshot()[0]:
                    self._job_forward_to_worker(job, random_node.get_uid())
                else:
                    # schedule locally
                    job.set_transmission_next_action(Job.TransmissionAction.NODE_TO_CLIENT)
                    self._job_schedule(job)

                if DEBUG:
                    Log.mdebug(f"{MODULE}#{self._uid}",
                               f"{job} probe executed: forwarded_to={job.get_forwarded_to()} now={self._env.now:.4f} queued_jobs={self._queued_probe_jobs.level}")

            except Exception as e:
                # if DEBUG:
                if self._currently_executing_job is not None:
                    Log.merr(f"{MODULE}#{self._uid}",
                             f"Node#{self.get_uid()} job probe executor interrupted, currently executing job={self._currently_executing_job} e={e}")
                else:
                    Log.merr(f"{MODULE}#{self._uid}", f"Node#{self.get_uid()} job probe executor interrupted, e={e}")

    def _process_job_transmission(self):
        """Process which simulates a job transmission"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"job transmission process started")

        def actual_time(delay, sigma):
            if self._distribution_network_latency == Node.DistributionNetworkLatency.GAUSSIAN:
                return random.gauss(delay, sigma)
            return delay

        def time_to_wait(job, speed):
            return job.get_payload_size() * 8 / speed

        while True:
            # print(self._queued_transmission_jobs.level)
            try:
                # wait for a job
                yield self._queued_transmission_jobs.get(1)
                job = self._jobs_transmission_list.pop(0)
                next_action = job.get_transmission_next_action()

                if DEBUG:
                    Log.mdebug(f"{MODULE}#{self._uid}",
                               f"_process_job_transmission: start transmitting job={job}, action={next_action}")
                tt = 0
                if next_action == Job.TransmissionAction.CLIENT_TO_SCHEDULER:
                    # job is sent from the client to the node
                    tt = actual_time(time_to_wait(job, self._net_speed_client_scheduler_mbits),
                                     self._distribution_network_forwarding_sigma)
                    yield self._env.timeout(tt)
                    # execute the first decision when arrived
                    self._job_first_dispatching(job)

                elif next_action == Job.TransmissionAction.SCHEDULER_TO_WORKER:
                    # job is transmitted from the scheduler to the worker
                    tt = actual_time(time_to_wait(job, self._net_speed_scheduler_worker_mbits),
                                     self._distribution_network_forwarding_sigma)
                    yield self._env.timeout(tt)
                    # set the next action
                    job.set_transmission_next_action(Job.TransmissionAction.WORKER_TO_SCHEDULER)
                    # schedule job to the worker
                    self._service_discovery.get_node_by_uid(job.get_forwarded_to_node_id()).schedule_job(job)

                elif next_action == Job.TransmissionAction.SCHEDULER_TO_CLOUD:
                    tt = actual_time(time_to_wait(job, self._net_speed_scheduler_cloud_mbits),
                                     self._distribution_network_latency_cloud_sigma)
                    # job is transmitted from the scheduler to the worker
                    yield self._env.timeout(tt)
                    # set the next action
                    job.set_transmission_next_action(Job.TransmissionAction.CLOUD_TO_SCHEDULER)
                    # schedule job to the worker
                    self._service_discovery.get_node_cloud().schedule_job(job)

                elif next_action == Job.TransmissionAction.SCHEDULER_TO_CLUSTER:
                    tt = actual_time(time_to_wait(job, self._net_speed_scheduler_scheduler_mbits),
                                     self._distribution_network_forwarding_sigma)
                    # job is transmitted from the scheduler to the worker
                    yield self._env.timeout(tt)
                    # retrieve the scheduler node of the cluster
                    cluster_scheduler_node = self._service_discovery.get_node_scheduler_for_cluster_id(
                        job.get_forwarded_to_cluster_id())
                    # next action
                    # job.set_transmission_next_action(Job.TransmissionAction.CLUSTER_TO_SCHEDULER)
                    # schedule the job there
                    cluster_scheduler_node.schedule_job(job)

                elif next_action == Job.TransmissionAction.CLOUD_TO_SCHEDULER or next_action == Job.TransmissionAction.WORKER_TO_SCHEDULER:
                    speed = self._net_speed_scheduler_cloud_mbits if next_action == Job.TransmissionAction.CLOUD_TO_SCHEDULER else self._net_speed_scheduler_worker_mbits
                    sigma = self._distribution_network_forwarding_sigma
                    # job is transmitted from the scheduler to the worker
                    yield self._env.timeout(actual_time(time_to_wait(job, speed), sigma))

                    # this means that the job comes from another cluster
                    if job.get_forwarded_to_cluster_id() > -1:
                        if job.get_forwarded_to_cluster_id() != self._node_belong_to_cluster:
                            Log.merr(f"{MODULE}#{self._uid}",
                                     f"job.originator={job.get_originator_node_uid()}, job.actions={job.get_transmission_actions_list()}")
                            raise RuntimeError(
                                f"This job should not be here! job.get_forwarded_to_cluster_id()={job.get_forwarded_to_cluster_id()}, self._node_belong_to_cluster={self._node_belong_to_cluster}")

                        # return the job to the original scheduler
                        job.set_transmission_next_action(Job.TransmissionAction.CLUSTER_TO_SCHEDULER)
                        self._job_transmit(job)
                    else:
                        # next step: return client
                        self.return_job_to_client(job)

                elif next_action == Job.TransmissionAction.SCHEDULER_TO_CLIENT:
                    # job returned to client, done
                    time_to_wait_s = actual_time(time_to_wait(job, self._net_speed_client_scheduler_mbits), self._distribution_network_forwarding_sigma)
                    yield self._env.timeout(time_to_wait_s)

                    # check if actions are correct
                    if DEBUG:
                        Log.mdebug(f"{MODULE}#{self._uid}", f"{job} returned to client: {job.get_transmission_actions_list()}")

                    if self._actions_space == Node.ActionsSpace.ONLY_WORKERS or self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
                        if len(job.get_transmission_actions_list()) not in [2, 4]:
                            raise RuntimeError(
                                f"Number of transmission actions is invalid is {job.get_transmission_actions_list()} job={job}")
                    if self._actions_space == Node.ActionsSpace.OTHER_CLUSTERS:
                        if len(job.get_transmission_actions_list()) not in [2, 4, 6]:
                            raise RuntimeError(
                                f"Number of transmission actions is invalid is {len(job.get_transmission_actions_list())}: {job.get_transmission_actions_list()} jobf={job.get_forwarded_to_cluster_id()}")

                    job.a_done()
                    self._service_data_storage.done_job(job, self._get_reward(job))

                elif next_action == Job.TransmissionAction.CLUSTER_TO_SCHEDULER:
                    # simulate the return to the client
                    originator_node = self._service_discovery.get_node_by_uid(job.get_originator_node_uid())
                    originator_node.return_job_to_client(job)

                else:
                    Log.merr(f"{MODULE}#{self._uid}", f"_process_job_transmission: {job} action not valid")

                self._time_round_total += job.get_total_time()

                if DEBUG:
                    Log.mdebug(f"{MODULE}#{self._uid}",
                               f"_process_job_transmission: end "
                               f"transmitted job={job}, next_action={job.get_transmission_next_action()}, level={self._queued_transmission_jobs.level}")

                self._time_round_transmission += tt
                self._time_energy_transmission += tt

            except Exception as e:
                traceback.print_exc()
                Log.merr(f"{MODULE}#{self._uid}", f"Node#{self.get_uid()} job transmission process interrupted, e={e}")
                exit(1)

    def _process_jobs_generator(self, job_arrival_type: Node.JobArrivalType, job_index: int):
        """Process which generates jobs to dispatch"""

        # checks
        if self._service_discovery is None:
            raise ValueError("Discovery service cannot be None.")

        if job_arrival_type == Node.JobArrivalType.PERIODIC:
            rate_l = self._job_periodic_rates_fps[job_index]
            sigma = self._job_periodic_arrival_time_std_devs[job_index]
            # job_percentages = self._job_periodic_percentages
            job_duration = self._job_periodic_durations[job_index]
            job_duration_std_dev = self._job_periodic_duration_std_devs[job_index]
            job_deadline = self._job_periodic_deadlines[job_index]
            job_payload_size = self._job_periodic_payload_sizes_mbytes[job_index]
            desired_fps = self._job_periodic_desired_rates_fps[job_index]
            desired_fps_min = self._job_periodic_desired_rates_fps_min[job_index]
            desired_fps_max = self._job_periodic_desired_rates_fps_max[job_index]
            job_type_id = job_index

        elif job_arrival_type == Node.JobArrivalType.EXPONENTIAL:
            rate_l = self._job_exponential_rates_fps[job_index]
            # sigma = self._job_exponential_arrival_time_std_devs[job_index]
            # job_percentages = self._job_exponential_percentages
            job_duration = self._job_exponential_durations[job_index]
            job_duration_std_dev = self._job_exponential_duration_std_devs[job_index]
            job_deadline = self._job_exponential_deadlines[job_index]
            job_payload_size = self._job_exponential_payload_sizes_mbytes[job_index]
            desired_fps = self._job_exponential_desired_rates_fps[job_index]
            desired_fps_min = self._job_exponential_desired_rates_fps_min[job_index]
            desired_fps_max = self._job_exponential_desired_rates_fps_max[job_index]
            job_type_id = self._job_periodic_types + job_index

        else:
            raise RuntimeError(f"job type is not valid: job_arrival_type={job_arrival_type}, job_index={job_index}")

        Log.minfo(self._module(), f"init: _process_jobs_generator, job_arrival_type={job_arrival_type}, job_index={job_index}")

        while True:
            # simulate job arrivals
            try:
                # draw time to wait
                if job_arrival_type == Node.JobArrivalType.PERIODIC:
                    time_to_wait = random.gauss(1 / rate_l, sigma)
                    while time_to_wait <= 0 or 1 / time_to_wait > rate_l:
                        time_to_wait = random.gauss(1 / rate_l, sigma)
                elif job_arrival_type == Node.JobArrivalType.EXPONENTIAL:
                    time_to_wait = random.expovariate(rate_l)
                    while time_to_wait <= 0:
                        time_to_wait = random.expovariate(rate_l)

                # wait
                yield self._env.timeout(time_to_wait)

                self._total_jobs += 1

                # generate a job
                job = Job(self._env, node_uid=self.get_uid(), uid=self._total_jobs, end_clb=self._clb_job_end,
                          eps=self._epsilon, payload_size_mbytes=job_payload_size,
                          duration=job_duration, episode=self._current_episode_number,
                          duration_std_dev=job_duration_std_dev,
                          deadline=job_deadline, fps_desired=desired_fps, fps_tolerance_max=desired_fps_max,
                          fps_tolerance_min=desired_fps_min,
                          job_type=job_type_id)

                if DEBUG:
                    Log.mdebug(self._module(), f"_process_jobs_generator: generated job of arrival type = {job_arrival_type.name}, "
                               f"job_type={job.get_type()} total_jobs={self._total_jobs}, "
                               f"episode={self._current_episode_number}, now={self._env.now}")
                    Log.mdebug(self._module(),
                               f"_process_jobs_generator: generated j={job}, now={self._env.now}, "
                               f"episode={self._current_episode_number}, rate_l={rate_l}, job_type_id={job_type_id}")

                # append the job to the backlog list
                self._scheduled_jobs.append(job)

                # check if episode ended
                episode_end = (self._total_jobs - self._last_episode_end_at) % self._episode_length == 0
                if episode_end:
                    if DEBUG:
                        Log.mdebug(self._module(),
                                   f"_process_jobs_generator: j=#{job}, episode #{self._current_episode_number}, limit={episode_end}")

                    job.set_last_of_episode(True)

                    self._current_episode_number += 1
                    self._last_episode_end_at = self._total_jobs

                    # increase epsilon
                    if self._eps_dynamic and self._epsilon > self._epsilon_min:
                        self._epsilon *= self._epsilon_decay

                # add to the transmission queue
                self._job_transmit(job)

            except simpy.Interrupt:
                Log.mwarn(self._module(), "Job generator interrupted")
                # raise RuntimeError(self._module() + ": Interrupted")
                exit(1)

    def _process_logger_impl(self):
        """Process used for logging values at specific times"""

        if self._session_learning_type is not Node.LearningType.D_SARSA:
            return

        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"_process_logger started")

        while True:
            try:
                # wait 1 second
                yield self._env.timeout(100)

                # log q value every second
                # def log_state_data(state, state_string):
                #    q_values = [self._d_sarsa_q(state, a) for a in self._get_actions()]
                #     for action, value in enumerate(q_values):
                #         self._service_data_storage.log_q_value_at_time(int(self._env.now), state_string, self._uid,
                # action, value)

                # log to db
                # all_possible_states = self._get_all_possible_states_str()
                # for state in all_possible_states:
                #     log_state_data([int(c) for c in state], state)


            except Exception as e:
                traceback.print_exc()
                Log.merr(f"{MODULE}#{self._uid}", f"Node#{self.get_uid()} _process_logger interrupted, e={e}")
                exit(1)

    def _process_save_loads(self):
        """This process updates the loads of the other worker every 1 second"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"_process_save_loads started")

        while True:
            try:
                yield self._env.timeout(1)

                loads = []

                # retrieve loads
                for node in self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster):
                    # log.Log.mdebug(f"{MODULE}#{self._uid}", f"Load of node {node.get_uid()}#{node.get_cluster_id(
                    # )}#{node.get_type()} is {node.get_current_load()}")
                    loads.append(node.get_current_load())

                self._loads_cluster = loads

                # log.Log.mdebug(f"{MODULE}#{self._uid}", f"Loads snapshot is {self._loads_cluster}")

                # save current load
                self._loads_our_history.append(self.get_current_load())
                if len(self._loads_our_history) > 10:
                    self._loads_our_history.pop(0)

            except simpy.Interrupt:
                Log.mwarn(self._module(), "Job generator interrupted")
                # raise RuntimeError(self._module() + ": Interrupted")
                exit(1)

    def _process_energy(self):
        if DEBUG:
            Log.mdebug(f"{self._module()}", f"_process_energy: started")

        while True:
            try:
                yield self._env.timeout(1)

                # discharge the battery
                to_discharge_ws = self._compute_battery_discharge(
                    1.0,
                    self._time_energy_execution,
                    self._time_energy_transmission)
      
                to_charge_ws = 0.0
                # recharge according to the solar model
                if self._solar_power_model is not None:
                    power_at_now = self._solar_power_model.get_watt_power_at(self._env.now)
                    to_charge_ws = power_at_now / 3600.0

                # reset
                self._time_energy_execution = .0
                self._time_energy_transmission = .0

                if not self._die_is_died and self.get_battery_residual_percentage() <= 0.0:
                    Log.minfo(self._module(),
                              f"_process_round: ep. ended, residual_battery_p={self.get_battery_residual_percentage()}")
                    workers = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
                    batteries = []
                    for worker in workers:
                        batteries.append(worker.get_battery_residual_wh())
                    self._die_is_died = True
                    self._service_data_storage.log_end_battery(self._env.now, self.get_uid(), max(batteries))
                    break
                
                if self._die_is_died and self.get_battery_residual_percentage() > 10.0:
                    self._die_is_died = False

                net_energy = to_charge_ws - to_discharge_ws
                self._metric_round_energy_list.append(to_discharge_ws)

                self._metric_round_energy_net += net_energy
                self._metric_round_energy_consumed += to_discharge_ws
                self._metric_round_energy_gained += to_charge_ws

                # update battery
                self._battery_current_capacity_wh += net_energy

                # check if maximum battery exceeds
                if self._battery_current_capacity_wh > self._battery_total_capacity_wh:
                    self._battery_current_capacity_wh = self._battery_total_capacity_wh

                # negative battery falls to zero
                if self._battery_current_capacity_wh < 0:
                    self._battery_current_capacity_wh = 0.0

                # check if the node must go idle by using hysteresis
                if self._energy_to_idle_thresholds_enabled:
                    if self._battery_current_capacity_wh >= self._energy_to_idle_high_threshold:
                        self._is_idle = False
                    if self._battery_current_capacity_wh <= self._energy_to_idle_low_threshold:
                        self._is_idle = True

                if DEBUG:
                    Log.mdebug(self._module(), f"_process_round: "
                                               f"t={self._env.now} "
                                               f"current_battery={self._battery_current_capacity_wh} "
                                               f"to_discharge_ws={to_discharge_ws} "
                                               f"to_charge_ws={to_charge_ws}")

            except Exception as e:
                traceback.print_tb(e.__traceback__)
                Log.mfatal(f"{self._module()}", f"_process_energy: interrupted, now={self._env.now} e={e}")

    def _process_die_simulator(self):
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"_process_save_loads started")

        # while True:
        try:
            yield self._env.timeout(self._die_after_seconds)

            # die
            self._die_is_died = True

            # back to life
            yield self._env.timeout(self._die_duration)

            self._die_is_died = False

        except simpy.Interrupt:
            Log.mwarn(self._module(), "Job generator interrupted")
            # raise RuntimeError(self._module() + ": Interrupted")
            exit(1)

    def _job_forward_to_worker(self, job, to_node: Node):
        job.set_transmission_next_action(Job.TransmissionAction.SCHEDULER_TO_WORKER)
        job.a_forwarded(to_node.get_uid())
        self._job_transmit(job)

    def _job_forward_to_cluster_index(self, job, to_cluster_index):
        if to_cluster_index < 0 or to_cluster_index > self._service_discovery.get_clusters_count() - 1:
            raise RuntimeError(f"Bad cluster index={to_cluster_index}")

        job.set_transmission_next_action(Job.TransmissionAction.SCHEDULER_TO_CLUSTER)

        # retrieve cluster id from index
        to_cluster_uid = self._service_discovery.get_cluster_uid_from_index(self._node_belong_to_cluster,
                                                                            to_cluster_index)
        # retrieve the scheduler for that cluster
        scheduler_node = self._service_discovery.get_node_scheduler_for_cluster_id(to_cluster_uid)

        # forward
        job.a_forwarded(scheduler_node.get_uid(), to_cluster_id=to_cluster_uid)

        self._job_transmit(job)

    def _job_forward_to_cloud(self, job):
        job.set_transmission_next_action(Job.TransmissionAction.SCHEDULER_TO_CLOUD)
        job.a_forwarded(to_cloud=True)
        self._job_transmit(job)

    def _job_probe(self, job):
        """Schedule a job for probing"""
        job.a_probing()
        # add the job to probing queue
        self._jobs_probing_list.append(job)
        self._queued_probe_jobs.put(1)

    def _job_transmit(self, job):
        """Schedule a job for transmission, direction is written in job.next_action"""
        self._jobs_transmission_list.append(job)
        self._queued_transmission_jobs.put(1)

    def _job_schedule(self, job):
        """Schedule a job internally in the node"""
        # if died, always reject
        if self._die_is_died:
            self._job_reject(job)
            return False

        # if node is full loaded
        if self.get_current_load() >= self._max_jobs_in_queue:
            self._job_reject(job)
            return False

        self._job_accept(job)
        return True

    def _job_accept(self, job: Job):
        self._jobs_list.append(job)
        self._queued_jobs.put(1)

        job.a_in_queue()

    def _job_reject(self, job):
        job.a_rejected()

        if job.get_forwarded_to_cluster_id() == -1:
            # job.set_transmission_next_action(Job.TransmissionAction.SCHEDULER_TO_CLIENT)
            self.return_job_to_client(job)
        else:
            job.set_transmission_next_action(Job.TransmissionAction.CLUSTER_TO_SCHEDULER)
            self._job_transmit(job)

    def _job_first_dispatching(self, job):
        """Execute the first dispatching action after the job arrived"""
        # get state snapshot
        state = self._get_state_representation(job)

        # make the decision
        chosen_action = self._act(state, job)

        job.set_reward_batteries(self._get_reward_battery(action=chosen_action))

        job.save_state_snapshot(state)
        job.save_action(chosen_action, self._actions_space != Node.ActionsSpace.OTHER_CLUSTERS)

        # mark as dispatched
        job.a_dispatched()

        # act
        self._act_execute(chosen_action, job)

    #
    # Core
    #

    def _act(self, state, job: Job) -> int or None:
        action = None
        if self._session_learning_type == Node.LearningType.NO_LEARNING:
            action = self._act_no_learning(state, job)
        elif self._session_learning_type == Node.LearningType.Q_DNN:
            raise RuntimeError("Not Implemented")
            # return self._act_dnnq(state, job)
        elif self._session_learning_type == Node.LearningType.Q_TABLE:
            raise RuntimeError("Not Implemented")
            # return self._act_q_table(state, job)
        elif self._session_learning_type == Node.LearningType.D_SARSA:
            action = self._act_d_sarsa(state, job)
        return action

    def _get_actions(self):
        """Retrieve all the possible actions in the form of array of numbers"""
        nodes_workers = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        if self._actions_space is Node.ActionsSpace.ONLY_WORKERS:
            # actions space = [REJECT, 0, 1, 2, 3, 4, 5, ...]
            return [i for i in range(1 + len(nodes_workers))]
        elif self._actions_space is Node.ActionsSpace.WORKERS_OR_CLOUD:
            # actions space = [REJECT, CLOUD, 1, 2, 3, 4, 5, ...]
            return [i for i in range(2 + len(nodes_workers))]
        elif self._actions_space is Node.ActionsSpace.OTHER_CLUSTERS:
            # actions space = [REJECT, CLOUD, W1, W2, W3, C1, C2, ...]
            return [i for i in range(2 + len(nodes_workers) + self._service_discovery.get_clusters_count() - 1)]
        else:
            Log.merr(f"{MODULE}#{self._uid}", "ActionsSpace not valid")
            raise RuntimeError("ActionsSpace not valid")

    def _get_possible_actions(self, job: Job, state: List[int]):
        """Returns the possible actions given the state"""
        all_actions = self._get_actions()
        n_workers = self._service_discovery.get_workers_in_cluster_count(self._node_belong_to_cluster)
        # possible_actions = all_actions

        # if other cluster forwarding is enabled, the job cannot be forwarded to another cluster if already forwarded
        if self._actions_space == Node.ActionsSpace.OTHER_CLUSTERS:
            if job.get_forwarded_to_cluster_id() > -1:
                if job.get_forwarded_to_cluster_id() == self._node_belong_to_cluster:
                    possible_actions = all_actions[1:1 + n_workers]
                else:
                    raise RuntimeError(
                        f"Job arrived here to wrong scheduler {job.get_forwarded_to_cluster_id()} != {self._node_belong_to_cluster}")
            else:
                # reject, cloud, available workers and other clusters
                possible_actions = [all_actions[0], all_actions[1]]
                # filter available nodes
                for i in all_actions[2:2 + n_workers]:
                    if sum(self._loads_cluster[i - 2]) < self._max_jobs_in_queue:
                        possible_actions.append(i)
                # add all clusters
                for i in all_actions[2 + n_workers:]:
                    possible_actions.append(i)
        elif self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
            possible_actions = [all_actions[0]]
            for i in all_actions[1:]:
                if sum(self._loads_cluster[i - 1]) < self._max_jobs_in_queue:
                    possible_actions.append(i)
        elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
            possible_actions = [all_actions[0], all_actions[1]]
            for i in all_actions[2:]:
                if sum(self._loads_cluster[i - 2]) < self._max_jobs_in_queue:
                    possible_actions.append(i)
        else:
            raise RuntimeError("Actions space is not supported")

        return possible_actions

    def _are_no_scheduling_actions(self, job: Job, state: List[int]) -> bool:
        """Returns true if the only action is to reject the job"""
        possible_actions = self._get_possible_actions(job, state)

        if len(possible_actions) == 1 and possible_actions[0] == self._action_size - 1:
            return True
        return False

    def _act_execute_old(self, action, job):
        """Execute the given action for the given job"""
        if action == Node.ACTION_EXECUTE_LOCALLY:
            self._job_schedule(job)

        if action == Node.ACTION_PROBE:
            # probe 1 random node
            random_node = self._service_discovery.get_random_node(current_node_id=self.get_uid())  # type: Node
            # check the load
            if random_node.get_current_load() < job.get_state_snapshot()[0]:
                job.a_forwarded(random_node.get_uid())
                # schedule it there
                random_node.schedule_job(job)
            else:
                # schedule locally
                self._job_schedule(job)

    def _act_execute(self, action: int, job: Job):
        """Execute the given action for the given job. See _get_actions for actions distribution"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}",
                       f"_act_execute: job={job}, action={action}, action_space={self._actions_space}")

        if action < 0:
            raise RuntimeError("Action is negative")

        worker_nodes = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        worker_nodes_n = len(worker_nodes)

        # update the internal representation of the state
        if job.get_forwarded_to_cluster_id() == -1:
            self._update_state(action, job, job_leaving=False)

        if self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
            if action == 0:
                return self._job_reject(job)
            else:
                return self._job_forward_to_worker(job, worker_nodes[action - 1])

        elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
            if action == 0:
                return self._job_reject(job)
            elif action == 1:
                return self._job_forward_to_cloud(job)
            else:
                return self._job_forward_to_worker(job, worker_nodes[action - 2])

        elif self._actions_space == Node.ActionsSpace.OTHER_CLUSTERS:
            if action == 0:
                return self._job_reject(job)
            elif action == 1:
                return self._job_forward_to_cloud(job)
            elif 1 < action < worker_nodes_n + 2:
                # log.Log.mdebug(MODULE, f"_act_execute: action={action}, worker_nodes={worker_nodes_n}")
                return self._job_forward_to_worker(job, worker_nodes[action - 2])
            else:
                cluster_index = action - 2 - worker_nodes_n
                return self._job_forward_to_cluster_index(job, cluster_index)
        else:
            Log.mfatal(f"{MODULE}#{self._uid}", f"_act_execute: action space {self._actions_space} is not valid")
            sys.exit(1)

    #
    # Core -> No learning
    #

    def _no_learning_log_episode(self):
        """Start the ordered memorization and finally the replay dnnq training"""
        # check if replay can start
        if not self._can_replay_start():
            return

        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}",
                       f"_memorize_and_replay_episode: started, len(self._scheduled_jobs)={len(self._scheduled_jobs)}")

        episode = self._scheduled_jobs[0].get_episode()
        episode_jobs = 0
        eps = self._scheduled_jobs[0].get_eps()
        episode_cumulative_reward = 0.0
        i = 0

        # memorize all the experience in order of job scheduling
        while i < len(self._scheduled_jobs) and self._scheduled_jobs[i].get_episode() == episode:
            self._total_processed_job += 1
            episode_jobs += 1

            job = self._scheduled_jobs.pop(i)
            state = job.get_state_snapshot()
            action = job.get_action(0)

            episode_cumulative_reward += self._get_reward(job)

            # if job.is_last_of_episode():
            #     reward = episode_cumulative_reward
            # else:
            # reward = self._get_reward(job)

        if self._logging_info:
            Log.minfo(self._module(), f"episode={episode} e={eps:.2f} score={episode_cumulative_reward} jobs={episode_jobs}")
            Log.minfo(self._module(), f"processed_jobs={self._total_processed_job} generated_jobs={self._total_jobs} "
                                      f"cur_episode={self._current_episode_number} "
                                      f"diff_episode={self._current_episode_number - episode} now={self._env.now}")

        # if self._total_processed_job > self._batch_size:
        self._log_episode_data(episode, eps, episode_cumulative_reward, self._total_processed_job, 0.0, 0.0, 0.0)

    def _act_no_learning(self, state, job):
        """Returns an action choosing from exploration and exploitation"""
        action = None
        # load = self._get_total_load_from_state(state)
        possible_actions = self._get_possible_actions(job, state)
        workers_in_cluster_count = self._service_discovery.get_workers_in_cluster_count(self._node_belong_to_cluster)

        if self._session_no_learning_policy == Node.NoLearningPolicy.RANDOM:
            action = possible_actions[random.randint(0, len(possible_actions) - 1)]

        elif self._session_no_learning_policy == Node.NoLearningPolicy.ROUND_ROBIN:
            action = self._round_robin_counter
            self._round_robin_counter = (self._round_robin_counter + 1) % len(possible_actions)

        elif self._session_no_learning_policy == Node.NoLearningPolicy.LEAST_LOADED_AWARE_CLOUD:
            if self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
                # longest jobs to cloud
                if job.get_type() == self._job_periodic_types + self._job_exponential_types - 1:
                    return 1  # 0=reject, 1=cloud, >1 workers
                else:
                    # check if only rejection and cloud is available
                    if len(possible_actions) == 2:
                        return 0  # reject
                    # others in the least loaded but first one
                    min_v = self._max_jobs_in_queue * 1000
                    min_i = 0
                    found = False
                    for action in possible_actions[2:]:
                        wkr = self._service_discovery.get_worker_in_cluster_by_index(self._node_belong_to_cluster,
                                                                                     action - 2)
                        if sum(self._loads_cluster[action - 2]) < min_v and not wkr.is_died():
                            min_v = sum(self._loads_cluster[action - 2])
                            min_i = action
                            found = True
                    if not found:
                        return 0  # reject
                        # raise RuntimeError("Cannot find min: ")
                    action = min_i
            else:
                raise RuntimeError("Action space not valid")

        elif self._session_no_learning_policy == Node.NoLearningPolicy.LEAST_LOADED_AWARE:
            if self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
                # check if only rejection is available
                if len(possible_actions) == 1:
                    return 0  # reject

                # find the least loaded node, pick the first (faster)
                min_v = self._max_jobs_in_queue * 1000
                min_i = 0
                found = False
                for action in possible_actions[1:]:
                    wkr = self._service_discovery.get_worker_in_cluster_by_index(self._node_belong_to_cluster, action - 1)
                    if sum(self._loads_cluster[action - 1]) < min_v and not wkr.is_died():
                        min_v = sum(self._loads_cluster[action - 1])
                        min_i = action
                        found = True
                if not found:
                    raise RuntimeError("Cannot find min")
                action = min_i
            else:
                raise RuntimeError("Action space not valid")

        elif self._session_no_learning_policy == Node.NoLearningPolicy.LEAST_LOADED_NOT_AWARE:

            if self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
                min_v = self._max_jobs_in_queue + 1
                min_i = 0
                for action in possible_actions[1:]:
                    wkr = self._service_discovery.get_worker_in_cluster_by_index(self._node_belong_to_cluster, action - 1)
                    if sum(self._loads_cluster[action - 1]) < min_v and not wkr.is_died():
                        min_v = sum(self._loads_cluster[action - 1])
                        min_i = action
                action = min_i
                
                if action == 0:
                    return 0
                
                count = 0
                i_arr = []
                for i, action_en in enumerate(possible_actions):
                    if sum(self._loads_cluster[action - 1]) == min_i:
                        count += 1
                        i_arr.append(i)
                if count > 0:
                    chosen = i_arr[random.randint(0, len(i_arr) - 1)]
                    action = possible_actions[chosen]

            elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
                min_v = self._max_jobs_in_queue + 1
                min_i = 0
                for action in possible_actions[2:]:
                    wkr = self._service_discovery.get_worker_in_cluster_by_index(self._node_belong_to_cluster, action - 2)
                    if sum(self._loads_cluster[action - 2]) < min_v and not wkr.is_died():
                        min_v = sum(self._loads_cluster[action - 2])
                        min_i = action
                action = min_i
                count = 0
                i_arr = []
                for i, action_en in enumerate(possible_actions[2:]):
                    if sum(self._loads_cluster[action - 2]) == min_i:
                        count += 1
                        i_arr.append(i)
                if count > 0:
                    chosen = i_arr[random.randint(0, len(i_arr) - 1)]
                    action = possible_actions[2:][chosen]
                # Log.mdebug(MODULE, f"self._loads_cluster={self._loads_cluster}, min_v={min_v}, min_i={min_i - 1}")
            else:
                action = 0
            
        elif self._session_no_learning_policy == Node.NoLearningPolicy.MAXIMUM_LIFESPANE:
            lifespans = []
            worker_nodes = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
            for worker in worker_nodes:
                lifespans.append(worker.get_lifespan())
                
            if self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
                action = lifespans.index(max(lifespans)) + 1
            elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
                action = lifespans.index(max(lifespans)) + 2
            else:
                raise RuntimeError("Action space not supported")

        else:
            Log.merr(f"{MODULE}#{self._uid}", f"Policy {self._session_no_learning_policy} not implemented")
            raise RuntimeError("Policy not implemented")

        return action

    #
    # Core -> Learning -> Sarsa
    #

    def _init_d_sarsa(self):
        # only scheduler
        if self._node_type == Node.NodeType.SCHEDULER:
            if self._use_model_from_session_name is not None:
                model_f = open(
                    f"{Node.BASE_DIR_MODELS}/{self._session_learning_type}/{self._use_model_from_session_name}/Node#{self._uid}/d_sarsa.model",
                    "rb")
                self._value_function = pickle.load(model_f)
                model_f.close()
            else:
                self._value_function = DSPSarsaTiling(num_tilings=self._tiling_num_tilings, max_size=33554432,
                                                      alpha=self._sarsa_alpha, beta=self._sarsa_beta)

            Log.minfo(self._module(), f"_init_d_sarsa: {self._value_function}")
        else:
            Log.minfo(self._module(), f"_init_d_sarsa: skipped for worker")

    def _d_sarsa_q(self, state, action):
        """Return the Q(S, A, w) value"""
        return self._value_function.value(state + [action])

    def _d_sarsa_learn_episode(self, alpha: float = 1.0):
        """Learn from experience of an episode"""
        if self._node_type == Node.NodeType.WORKER:
            Log.mdebug(self._module(), f"node_uid={self.get_uid()}")
            raise RuntimeError("_d_sarsa_learn_episode called from worker node")

        # check if replay can start
        if not self._can_replay_start():
            return

        time_start = time.time()

        if DEBUG:
            Log.mdebug(self._module(),
                       f"_d_sarsa_learn_episode: started, len(self._scheduled_jobs)={len(self._scheduled_jobs)}")

        episode = self._scheduled_jobs[0].get_episode()
        episode_jobs = 0
        eps = self._scheduled_jobs[0].get_eps()
        episode_cumulative_reward = 0.0
        last_action = 0
        i = 0

        # process the first job
        job = self._scheduled_jobs.pop(i)
        state = job.get_state_snapshot()
        action = job.get_action(0)

        reward_fps = self._get_reward(job)
        reward_battery = job.get_reward_batteries()
        reward = alpha * reward_fps + (1.0 - alpha) * reward_battery

        episode_cumulative_reward += reward
        losses = []

        # save the job in backlog
        self._job_latest_processed[job.get_type()] = job
        # print(f"self._job_latest_processed[job.get_type()]={self._job_latest_processed[job.get_type()]}")

        self._total_processed_job += 1
        episode_jobs += 1

        # memorize all the experience in order of job scheduling
        while len(self._scheduled_jobs) > 0 and self._scheduled_jobs[0].get_episode() == episode:
            job = self._scheduled_jobs.pop(0)
            new_state = job.get_state_snapshot()
            new_action = job.get_action(0)  # if not job.is_rejected() else self._action_size - 1

            # save the job in backlog
            self._job_latest_processed[job.get_type()] = job

            current_full_state = state + [action]
            new_full_state = new_state + [new_action]

            # Log.mdebug(f"{MODULE}#{self.get_uid()}",
            #            f"current={current_full_state}, new={new_full_state}, reward={reward}")

            # update weights
            loss = self._value_function.learn(current_full_state, new_full_state, reward)
            losses.append(abs(loss))

            if DEBUG:
                Log.mdebug(self._module(),
                           f"_d_sarsa_learn_episode: ep={job.get_episode()} jid={job}: state={state}, action={action}, "
                           f"new_state={new_state}, reward={reward}")

            # save current state
            reward_fps = self._get_reward(job)
            reward_battery = job.get_reward_batteries()
            reward = alpha * reward_fps + (1 - alpha) * reward_battery
            
            state = new_state
            action = new_action

            # update counters
            self._total_processed_job += 1
            episode_jobs += 1
            episode_cumulative_reward += reward

        # if self._total_processed_job > self._batch_size:
        self._log_episode_data(episode, eps, episode_cumulative_reward, self._total_processed_job,
                               sum(losses) / float(len(losses)), 0.0, 0.0)

        elapsed = time.time() - time_start

        if self._logging_info:
            Log.minfo(self._module(),
                      f"episode={episode} e={eps:.2f} score={episode_cumulative_reward} jobs={episode_jobs} "
                      f"average_reward={self._value_function.stats()} "
                      f"processed_jobs={self._total_processed_job} generated_jobs={self._total_jobs} "
                      f"cur_episode={self._current_episode_number} diff_episode={self._current_episode_number - episode} "
                      f"now={self._env.now} elapsed={elapsed}")

    def _act_d_sarsa(self, state: List[int], job: Job):
        """Given the state as a list of integers and the job, take an action according epsilon and the Q(s,a,w) function"""
        possible_actions = self._get_possible_actions(job, state)

        if possible_actions is None or len(possible_actions) == 0:
            raise RuntimeError("actions is none")

        # check if we need to perform a random action
        if np.random.rand() <= self._epsilon:
            action = self._act_no_learning(state, job)
            # action = possible_actions[random.randrange(len(possible_actions))]
        else:
            values = [(a, self._d_sarsa_q(state, a)) for a in possible_actions]
            max_value = max(values, key=lambda item: item[1])[1]
            max_actions = [action_ for action_, value_ in values if value_ == max_value]

            if DEBUG:
                Log.mdebug(self._module(),
                           f"_act_d_sarsa: j={job} state={state} values={values}, max={max_value}, max_actions={max_actions}")
            # Log.mdebug(self._module(), f"_act_d_sarsa: jid={job.get_id()} weights={self._weights}")
            action = np.random.choice(max_actions)

        if DEBUG:
            Log.mdebug(self._module(),
                       f"_act_d_sarsa: j={job} ep={job.get_episode()} state={state} possible_actions={possible_actions} chosen_action={action}")

        return action

    #
    # Callbacks
    #

    def _clb_job_end(self, job):
        """Called when a job ends by the generator node, executed or rejected"""

        if self._node_type == Node.NodeType.WORKER:
            Log.mdebug(self._module(), f"node_uid={self.get_uid()}")
            raise RuntimeError("_clb_job_end called from worker node")

        if DEBUG:
            Log.mdebug(self._module(), f"_clb_job_end: j={job}, episode #{job.get_episode()}: "
                                       f"end: deadline={job.get_deadline()} total_time={job.get_total_time():.2f} "
                                       f"executed={job.is_executed()} rejected={job.is_rejected()} reward={self._get_reward(job)}")

        # update the internal representation of the state
        self._update_state(job.get_action(0), job, job_leaving=True)

        if len(job.get_state_updated()) == 1 or (len(job.get_state_updated()) != 2 and 2 <= job.get_action(0) <= len(
                self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster))):
            raise RuntimeError(
                f"Job state updated list is={job.get_state_updated()}, actions={job.get_actions()}, t_actions={job.get_transmission_actions_list()}")

        if self._session_learning_type == Node.LearningType.Q_DNN:
            # replay and memorize, if possible
            # self._dnnq_memorize_and_replay_episode()
            pass
        elif self._session_learning_type == Node.LearningType.Q_TABLE:
            # self._q_table_learn_episode()
            pass
        elif self._session_learning_type == Node.LearningType.D_SARSA:
            self._d_sarsa_learn_episode(alpha=self._reward_alpha)
        else:
            self._no_learning_log_episode()

    #
    # Utils
    #

    def _get_state_representation(self, arrived_job: Job) -> List[int]:
        """Returns the current state used for learning"""
        lifespans = []
        worker_nodes = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        for worker in worker_nodes:
            lifespans.append(worker.get_lifespan())

        min_lifespan = min(lifespans)
        if min_lifespan > 0:
            lifespans = [(ls - min_lifespan) for ls in lifespans]
            max_lifespan = max(lifespans)
            lifespans = [(ls / max_lifespan) for ls in lifespans]
        else:
            lifespans = [worker.get_battery_residual_wh() for worker in worker_nodes]
            
        if self._state_type == Node.StateType.ONLY_NUMBER:
            state = [arrived_job.get_type()]
            for loads in self._loads_cluster:
                state.append(sum(loads))          
            for wh in lifespans:
                state.append(wh)
        elif self._state_type == Node.StateType.JOB_TYPE:
            state = [arrived_job.get_type()]
            for loads in self._loads_cluster:
                for load in loads:
                    state.append(load)
            for wh in lifespans:
                state.append(wh)
        else:
            Log.mfatal(MODULE, "StateRepresentation not implemented")
            sys.exit(1)

        return state

    def _is_episode_over(self, job: Job, state: List[int]):
        """Return if episode is over"""
        # return len(self._get_actions(job, state)) == 0
        return self._total_jobs % self._episode_length == 0

    def _can_replay_start(self):
        """Check if replay and memoization can start"""
        if DEBUG:
            Log.mdebug(self._module(), f"_can_replay_start: called, len(self._scheduled_jobs)={len(self._scheduled_jobs)}")

        if len(self._scheduled_jobs) == 0:
            return False

        last_job_executed = False
        episode = self._scheduled_jobs[0].get_episode()
        i = 0
        while i < len(self._scheduled_jobs) and self._scheduled_jobs[i].get_episode() == episode:
            job = self._scheduled_jobs[i]
            if self.get_uid() == 0:
                if DEBUG:
                    Log.mdebug(self._module(), f"_can_replay_start: id={job} ep={job.get_episode()} "
                                               f"is_done={job.is_done()} is_last={job.is_last_of_episode()} "
                                               f"rej={job.is_rejected()} action={job.get_action(0)} "
                                               f"total_time={job.get_total_time():.4f} forwarded_to_cloud={job.is_forwarded_to_cloud()} "
                                               f"forwarded_to_node={job.get_forwarded_to_node_id()} probing_time={job.get_probing_time():.4f} "
                                               f"now={self._env.now:.4f} time_gen={job._time_generated:.4f} "
                                               f"time_queued={job.get_queue_time():.4f} next_action={job.get_transmission_next_action()} "
                                               f"dispatched_time={job.get_dispatched_time():.4f} dispatched={job.is_dispatched()}")
            if not job.is_done():
                return False

            # check if the last job has been executed
            if job.is_last_of_episode():
                last_job_executed = True

            i += 1

        return last_job_executed

    def _get_reward_battery(self, action):
        def get_lifespan_reward(action_worker, worker_nodes):
            lifespans = []
            for worker in worker_nodes:
                lifespans.append(worker.get_lifespan())

            max_lifespan = max(lifespans)
            lifespans = [1 if i == max_lifespan else -0.5 for i in lifespans]
            return 2 * lifespans[action_worker]
        
        worker_nodes = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        """Get the reward for the battery"""
        if self._actions_space == Node.ActionsSpace.ONLY_WORKERS:
            if action == 0 or worker_nodes[action - 1].is_died():
                return -4
            else:
                return get_lifespan_reward(action_worker=action - 1, worker_nodes=worker_nodes)

        elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD:
            if action == 0:
                return -4
            elif action == 1:
                return 1
            elif worker_nodes[action - 2].is_died():
                return -4
            else:
                return get_lifespan_reward(action_worker=action - 2, worker_nodes=worker_nodes)

        else:
            Log.mfatal(f"{MODULE}#{self._uid}", f"_act_execute: action space {self._actions_space} is not valid")
            sys.exit(1)

    def _get_reward(self, job: Job):
        """Get the reward for the action or for the episode if it is over"""
        if not job.is_done():
            raise RuntimeError(
                "You are requesting the reward from a job that has not returned to the client, please check your code!")

        if self._node_type == Node.NodeType.WORKER:
            raise RuntimeError("Called _get_reward from a worker")

        # get latest processed
        # job_previous = self._job_latest_processed[job.get_type()]

        # if job_previous is None:
        #     return 1

        # lag = (job.get_generated_at() + job.get_total_time()) - (job_previous.get_generated_at() + job_previous.get_total_time())

        # no frame loss
        if job.is_succeed():
            # # frame loss
            # if lag < 0:
            #     return -2

            if job.is_meeting_fps_requirement():
                return 2
            elif job.is_over_fps_requirement():
                return 3
            elif job.is_less_fps_requirement():
                return -1
            else:
                raise RuntimeError("This is not possible")
        else:
            return -4

    def _get_nodes_state(self):
        return [node.get_current_load() for node in self._service_discovery.get_all_nodes()]

    def _update_state(self, action, job, job_leaving=False):
        """Update the internal state representation when job arrives"""
        workers = self._service_discovery.get_workers_in_cluster(self._node_belong_to_cluster)
        workers_n = len(workers)

        def check_state_value(action_used, states_array):
            for v in states_array:
                if v < 0 or v > self._max_jobs_in_queue + 10:
                    raise RuntimeError(f"state for action={action_used} is {states_array}")

        if self._actions_space == Node.ActionsSpace.ONLY_WORKERS and action > 0:  # is reject
            state_i = action - 1
        elif self._actions_space == Node.ActionsSpace.WORKERS_OR_CLOUD and action > 1:  # 0 is reject, 1 is cloud
            state_i = action - 2
        elif self._actions_space == Node.ActionsSpace.OTHER_CLUSTERS and 2 <= action <= workers_n + 1:
            state_i = action - 2
        else:
            # raise RuntimeError(f"Action space {self._actions_space} is not implemented, action={action}")
            # Log.mdebug(MODULE, f"_update_state: jid={job.get_uid()} action={action}, workers_n={workers_n}, job_leaving={job_leaving}")
            return False

        if state_i < 0:
            raise RuntimeError(f"state < 0, action={action}, self._actions_space={self._actions_space}")

        job.set_state_updated(action, job_leaving)

        self._loads_cluster[state_i][job.get_type()] += 1 if not job_leaving else -1
        check_state_value(state_i, self._loads_cluster[state_i])

        batteries = []

        # retrieve loads
        for node in workers:
            batteries.append(node.get_battery_residual_wh())

        self._batteries_cluster = batteries

        return True

    #
    # Logging
    #

    def _log_episode_data(self, episode, eps, score, total_jobs, loss, mse, mae):
        self._service_data_storage.done_episode(self._uid, episode, eps, score, total_jobs, loss, mse, mae)

    def _module(self):
        return f"{self.MODULE}#{self.get_uid()}"

    #
    # Exported methods
    #

    def set_service_discovery(self, service_discovery):
        self._service_discovery = service_discovery

    def set_service_data_storage(self, service_data_storage):
        self._service_data_storage = service_data_storage

    def is_idle(self):
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}",
                       f"is_idle: id={self.get_uid()} id={id(self)} idle={self._currently_executing_job is None} level={self._queued_jobs.level} "
                       f"list={len(self._jobs_list)} time={self._env.now:.4f} level_id={id(self._queued_jobs)}")
        return self._currently_executing_job is None and self._queued_jobs.level == 0 and len(self._jobs_list) == 0

    def schedule_job(self, job):
        """Schedule a job from external, without adding to the transmission queue, since only who sends is in charge
        of the transmission"""
        if DEBUG:
            Log.mdebug(f"{MODULE}#{self._uid}", f"schedule_job: scheduling j={job}")

        # scheduler behavior
        if self._node_type == Node.NodeType.SCHEDULER:
            if job.get_forwarded_to_cluster_id() != self._node_belong_to_cluster:
                raise RuntimeError(
                    "Cannot schedule a job in the scheduler node from the workers or client, or forwarding error")

            # dispatch
            return self._job_first_dispatching(job)

        return self._job_schedule(job)

    def return_job_to_client(self, job):
        """Re-add to transmission queue for re-trasmit the node to the client. This is called after a job is executed
        remotely"""
        job.set_transmission_next_action(Job.TransmissionAction.SCHEDULER_TO_CLIENT)
        scheduler_node = self._service_discovery.get_node_scheduler_for_cluster_id(self._node_belong_to_cluster)
        scheduler_node._job_transmit(job)

    def return_job_from_cloud(self, job):
        """Re-add to transmission queue for re-trasmit from cloud to scheduler. This is called after a job is executed
        in the cloud"""
        self._job_transmit(job)

    def is_running_job(self):
        return self._currently_executing_job is not None

    def get_current_load(self) -> int:
        """Retrieve the number of queued tasks"""
        return self._queued_jobs.level + 1 if self._currently_executing_job is not None else 0

    def get_uid(self):
        """Return the identifier number of the node"""
        return self._uid

    def get_job_list(self):
        return self._jobs_list

    def get_currently_executing_job(self):
        return self._currently_executing_job

    def get_max_queue_length(self):
        return self._max_jobs_in_queue

    def get_cluster_id(self):
        return self._node_belong_to_cluster

    def get_type(self) -> NodeType:
        return self._node_type

    def is_died(self):
        return self._die_is_died

    def get_models_dir(self):
        return self._DIR_MODELS

    def get_value_function(self):
        return self._value_function

    def get_learning_type(self):
        return self._session_learning_type

    # energy
    def _compute_battery_discharge_split(self, time_idle, time_execution, time_transmission) -> (float, float, float):
        discharge_idle = time_idle * (self._power_idle_w / 3600.0)
        discharge_execution = time_execution * (self._power_max_cpu_w / 3600.0)
        discharge_transmission = time_transmission * (self._power_max_transmission_w / 3600.0)

        return discharge_idle, discharge_execution, discharge_transmission

    def _compute_battery_discharge(self, time_idle, time_execution, time_transmission):
        d_idle, d_execution, d_transmission = self._compute_battery_discharge_split(time_idle, time_execution,
                                                                                    time_transmission)
        return d_idle + d_execution + d_transmission

    def get_battery_residual_wh(self) -> float:
        return self._battery_current_capacity_wh

    def get_battery_residual_percentage(self) -> float:
        return self._battery_current_capacity_wh / self._battery_total_capacity_wh

    def get_lifespan(self):
        if self.get_power_absorbed_round() <= 0.0:
            return 0.0
        return self._battery_current_capacity_wh / self.get_power_absorbed_round()

    def get_power_absorbed_round(self):
        discharge_ws = self._metric_round_energy_consumed  # Ws - watt/second
        return (3600.0 * discharge_ws) / self._round_time_s
    
    def get_power_gained_round(self):
        charge_ws = self._metric_round_energy_gained  # Ws - watt/second
        return (3600.0 * charge_ws) / self._round_time_s

    def _log_battery(self,batteries_cluster):
        self._service_data_storage.log_battery(self._env.now, self._uid, self._battery_current_capacity_wh, variance(batteries_cluster))
