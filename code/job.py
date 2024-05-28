#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from __future__ import annotations

import random
from enum import Enum
from functools import reduce
from typing import List

import simpy


class Job:
    MODULE = "Job"

    class TransmissionAction(Enum):
        CLIENT_TO_SCHEDULER = 0
        SCHEDULER_TO_WORKER = 1
        SCHEDULER_TO_CLOUD = 2
        SCHEDULER_TO_CLUSTER = 3
        SCHEDULER_TO_CLIENT = 5
        WORKER_TO_SCHEDULER = 6
        CLOUD_TO_SCHEDULER = 7
        CLUSTER_TO_SCHEDULER = 8

    class DurationType(Enum):
        FIXED = 0
        GAUSSIAN = 1
        EXPONENTIAL = 2

    def __init__(self, env: simpy.Environment, uid=0, node_uid=0, job_type=0, end_clb=None, duration=1.0,
                 duration_std_dev=0.0013, deadline=1.5, eps=0.0, payload_size_mbytes=0.1, episode=0, fps_desired=30,
                 fps_tolerance_max=30, fps_tolerance_min=25):
        """Create a new job"""
        self._uid = f"{node_uid}-{uid}"
        """Job identifier"""
        self._node_uid = node_uid
        """Job node generator identifier"""
        self._env = env
        """Simpy environment"""
        self._deadline = deadline  # s
        """Deadline of the job in seconds"""
        """If job is realtime"""
        self._job_duration = duration  # s
        """Average duration of a job"""
        self._job_duration_std_dev = duration_std_dev  # s
        """Standard deviation of a job"""
        self._payload_size = payload_size_mbytes
        """Size of the payload in mbytes"""

        self._transmission_next_action = Job.TransmissionAction.CLIENT_TO_SCHEDULER
        self._transmission_actions_list = [Job.TransmissionAction.CLIENT_TO_SCHEDULER]
        self._transmission_actions_list_timings = [0.0]

        # realtime check
        self._job_type = job_type

        # metrics
        self._metric_execution_time = 0.0  # s
        self._metric_queue_time = 0.0  # s

        # time snapshots
        self._time_generated = env.now
        self._time_queued = 0.0
        self._time_executed = 0.0
        self._time_done = 0.0
        self._time_forwarded = 0.0
        self._time_probing = 0.0
        self._time_rejected = 0.0
        self._time_dispatched = 0.0
        """Time at which"""

        self._forwarded_to_node_id = -1
        self._forwarded_to_cluster_id = -1
        self._forwarded_to_cloud = False

        # self._current_probing_time = 0.0  # s
        self._dispatched = False
        self._forwarded = False
        self._rejected = False
        self._executed = False
        self._done = False
        """If job path has ended"""

        # data to be logged
        self._state_snapshots = []  # type: List[List[int]]
        """The state of all nodes before dispatching"""
        self._actions = []
        """The node to which the job has been dispatched"""
        self._reward = 0.0  # type float
        """The reward for dispatching the job"""
        # callbacks
        self._end_clb = end_clb
        self._episode = episode
        """Episode to which the job belongs"""
        self._last = False
        """If it is the last job of the episode"""
        self._eps = eps
        """The eps value for the job"""

        self._fps_desired = fps_desired
        self._fps_tolerance_max = fps_tolerance_max
        self._fps_tolerance_min = fps_tolerance_min

        self._state_updated = []
        self._reward_batteries = 0.0

    def __str__(self):
        return f"Job#{self._uid}#N{self._node_uid}#E{self._episode}"

    def get_reward_batteries(self):
        return self._reward_batteries

    def set_reward_batteries(self, reward_batteries):
        self._reward_batteries = reward_batteries

    #
    # Actions
    #

    def a_dispatched(self):
        self._time_dispatched = self._env.now
        self._dispatched = True

    def a_in_queue(self):
        """Add time in queue"""
        self._time_queued = self._env.now

    def a_probing(self):
        """Mark when probing is started"""
        self._time_probing = self._env.now

    def a_executed(self, job_duration: float):
        self._metric_execution_time = job_duration
        self._executed = True

        self._time_executed = self._env.now

    def a_forwarded(self, to_node_uid=-1, to_cloud=False, to_cluster_id=-1):
        """Execute when a job is forwarded to another node"""
        if to_node_uid == -1 and to_cloud is False and to_cluster_id == -1:
            raise RuntimeError("You called the forwarded action with no parameters")

        self._time_forwarded = self._env.now
        self._forwarded_to_node_id = to_node_uid
        self._forwarded_to_cloud = to_cloud

        # update the cluster_id only once, this because a job can be forwarded to clusters
        if self._forwarded_to_cluster_id == -1:
            self._forwarded_to_cluster_id = to_cluster_id

    def a_rejected(self):
        self._rejected = True
        self._time_rejected = self._env.now

    def a_done(self):
        """Call this function when job path end, i.e. job output reached the client"""
        # safety checks
        if not self._executed and not self._rejected:
            raise RuntimeError(
                f"You declared done {self} without being executed or rejected, dispatched={self._dispatched} forwarded={self._forwarded}")
        if (self._rejected or (self._executed and self._forwarded)) and not (
                len(self._transmission_actions_list) in [2, 4, 6]):
            raise RuntimeError(
                f"Job rejected with bad action list: {self}, executed={self._executed}, forward={self._forwarded}, {self._transmission_actions_list}")

        self._done = True
        self._time_done = self._env.now

        if self._end_clb is not None:
            self._end_clb(self)

        # if self.get_total_time() > 0.1:
        #     print(self._transmission_actions_list)
        #     print(self._transmission_actions_list_timings)
        #     raise RuntimeError(f"TooMuch: {self.get_total_time()}")

    #
    # Exported
    #

    @staticmethod
    def compute_duration(job, job_duration_type, machine_speed: float):
        # compute the job duration given the machine speed
        job_duration = job.get_job_duration() * (1 / machine_speed)

        # choose the duration distribution
        if job_duration_type == Job.DurationType.GAUSSIAN:
            actual_job_duration = -1.0
            # loop until we get non-negative durations
            while actual_job_duration <= 0.0:
                actual_job_duration = random.gauss(job_duration, pow(job.get_job_duration_std_dev(), 2))
            return actual_job_duration
        elif job_duration_type == Job.DurationType.EXPONENTIAL:
            return random.expovariate(1 / job_duration)
        else:
            return job_duration

    def get_generated_at(self):
        return self._time_generated

    def get_executed_by(self):
        return self._action

    def get_transmission_next_action(self) -> TransmissionAction:
        return self._transmission_next_action

    def get_transmission_actions_list(self) -> List[TransmissionAction]:
        return self._transmission_actions_list

    def is_rejected(self):
        return self._rejected

    def is_forwarded(self):
        return self._forwarded

    def is_forwarded_to_cloud(self):
        return self._forwarded_to_cloud

    def get_forwarded_to_cluster_id(self):
        return self._forwarded_to_cluster_id

    def is_dispatched(self):
        return self._dispatched

    def is_over_deadline(self):
        return self.get_total_time() > self._deadline

    def is_meeting_fps_requirement(self):
        return self._fps_tolerance_min <= (1 / self.get_total_time()) <= self._fps_tolerance_max

    def is_less_fps_requirement(self):
        return (1 / self.get_total_time()) < self._fps_tolerance_min

    def is_over_fps_requirement(self):
        return (1 / self.get_total_time()) > self._fps_tolerance_max

    def is_succeed(self):
        return self.is_executed() and not self.is_rejected()

    def get_fps_desired(self):
        return self._fps_desired

    def get_fps_max(self):
        return self._fps_tolerance_max

    def get_fps_min(self):
        return self._fps_tolerance_min

    def get_type(self) -> int:
        return self._job_type

    def is_executed(self):
        """Get if job has been executed. Obviously, job rejected are not executed"""
        return self._executed

    def is_done(self):
        """Get if job has been executed. Obviously, job rejected are not executed"""
        return self._done

    def get_total_time(self) -> float:
        """Get the total elapsed time for executing the job"""
        # if not self._done:
        #    Log.mwarn(self.MODULE, "Requesting total time while job has not done")
        return self._time_done - self._time_generated

    def get_originator_node_uid(self):
        return self._node_uid

    def get_slack_time(self):
        return self._deadline - self.get_total_time()

    def get_forwarded_to_node_id(self):
        return self._forwarded_to_node_id

    def get_probing_time(self):
        return self._time_probing - self._time_generated

    def get_dispatched_time(self):
        return self._time_dispatched - self._time_generated

    def get_queue_time(self):
        return self._time_queued - self._time_generated

    def get_duration(self):
        """Get the programmed job duration"""
        return self._job_duration

    def get_deadline(self):
        return self._deadline

    def get_payload_size(self):
        return self._payload_size

    def get_episode(self):
        return self._episode

    def get_eps(self):
        return self._eps

    def get_uid(self):
        return self._uid

    def get_node_uid(self):
        return self._node_uid

    def is_last_of_episode(self):
        return self._last

    def set_transmission_next_action(self, action: Job.TransmissionAction):
        if action == self._transmission_actions_list[-1]:
            raise RuntimeError(f"You are setting the same action two times: list={self._transmission_actions_list}")

        self._transmission_next_action = action
        self._transmission_actions_list.append(action)
        self._transmission_actions_list_timings.append(self._env.now - self._time_generated)

    def set_episode(self, episode):
        self._episode = episode

    def set_last_of_episode(self, last):
        self._last = last

    def set_eps(self, eps):
        self._eps = eps

    def set_state_updated(self, action, leaving):
        self._state_updated.append(f"{action}-{leaving}")

    #
    # Time generators
    #

    def get_job_duration(self) -> float:
        return self._job_duration

    def get_job_duration_std_dev(self) -> float:
        return self._job_duration_std_dev

    def get_total_time_execution(self) -> float:
        return self._metric_execution_time

    def get_time_execution(self):
        return self._time_executed - self._time_generated

    #
    # DNN Data
    #

    def save_state_snapshot(self, snapshot: List[int]):
        self._state_snapshots.append(snapshot)

    def save_action(self, action, only_one_action=True):
        self._actions.append(action)

        # Log.mdebug("Job", f"saved action for {self.get_uid()}, action={action}")

        if only_one_action and len(self._actions) > 1:
            raise RuntimeError(
                f"Wrong number of actions: actions={self._actions}, list={self._transmission_actions_list}, rej={self._rejected}, exec={self._executed}, done={self._done}, dispatched={self._dispatched}")

    def get_state_snapshot(self) -> List[int]:
        return self._state_snapshots[0]

    def get_state_snapshot_str(self) -> List[int]:
        return reduce(lambda x, y: str(x) + str(y), self._state_snapshots[0])

    def get_action(self, i) -> int:
        """Get action at index"""
        return self._actions[i]

    def get_actions(self) -> List[int]:
        """Get action at index"""
        return self._actions

    def get_state_updated(self):
        return self._state_updated
