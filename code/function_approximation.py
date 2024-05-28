#  Real-time, adaptive and online scheduling for Edge-to-Cloud Continuum based on Reinforcement Learning
#  Copyright (c) 2024. Andrea Panceri <andrea.pancio00@gmail.com>
#
#   All rights reserved.

from typing import List

import numpy as np

from tilings import tiles, IHT


class DSPSarsaPolynomial:
    """Differential Semi-gradient Sarsa with polynomial approx."""

    def __init__(self, state_dim=4, polynomial_order=2, alpha=.1, beta=.1):
        self._alpha = alpha
        self._beta = beta
        self._state_dim = state_dim
        self._polynomial_order = polynomial_order
        self._n_features = pow(polynomial_order + 1, state_dim)

        self._average_reward = 0.0

        # construct the cij matrix
        self._weights = {}
        self._cij_matrix = []

        def gen_cij_matrix(array, values, position, max_position):
            if position == max_position:
                self._cij_matrix.append(array)
                self._weights[self._ats(array)] = 1
                return

            for v in values:
                gen_cij_matrix(array + [v], values, position + 1, max_position)

        gen_cij_matrix([], list(range(polynomial_order + 1)), 0, state_dim)

    def __str__(self):
        return f"DSPSarsaPolynomial(state_dim={self._state_dim}, polynomial_order={self._polynomial_order}, alpha={self._alpha}, beta={self._beta})"

    def _ats(self, state):
        """Convert array of ints to string"""
        out = ""
        for s in state:
            out += str(s)
        return out

    def _sta(self, string):
        """Convert string to array of int"""
        out = []
        for s in string:
            out += [int(s)]
        return out

    def value(self, state: List[int]):
        out = 0.0
        for i in range(self._n_features):
            # pick exponents from the cij matrix
            exponents = self._cij_matrix[i]
            # compute the feature value with the weight
            feature_value = self._weights[self._ats(exponents)]
            for j, e in enumerate(exponents):
                feature_value *= pow(state[j], e)
            out += feature_value
        return out

    def value_of_feature(self, state, fid: str):
        """Return the value of the specific feature and weight (w_i*x_i(s))"""
        exponents = self._sta(fid)
        # compute the feature value with the weight
        feature_value = self._weights[self._ats(exponents)]
        for j, e in enumerate(exponents):
            feature_value *= pow(state[j], e)
        return feature_value

    def learn(self, past_state, next_state, reward):
        delta = reward - self._average_reward + self.value(past_state) - self.value(next_state)
        self._average_reward += self._beta * delta

        # update weights
        for i in range(self._n_features):
            # pick exponents from the cij matrix
            exponents = self._cij_matrix[i]
            # compute the feature value with the weight
            feature_value = self.value_of_feature(past_state, self._ats(exponents))
            # update weight for that feature
            self._weights[self._ats(exponents)] += self._alpha * delta  # * feature_value

    def stats(self):
        """Returns an array of internal stats"""
        return [self._average_reward]


class DSPSarsaTiling:
    """Implement Differential Semi-gradient Sarsa with tiling"""

    def __init__(self, num_tilings=8, max_size=1024, alpha=.1, beta=.1):
        self._max_size = max_size
        self._num_of_tilings = num_tilings
        self._alpha = alpha
        self._beta = beta

        self._hash_table = IHT(self._max_size)
        self._weights = np.zeros(self._max_size)
        self._scale_state = self._num_of_tilings / 4

        self._average_reward = 0.0

    def __str__(self):
        return f"DSPSarsaTiling(num_tilings={self._num_of_tilings}, max_size={self._max_size}, alpha={self._alpha}, beta={self._beta})"

    def _get_active_tiles(self, state):
        action = state[-1]
        active_tiles = tiles(self._hash_table, self._num_of_tilings,
                             [s * self._scale_state for s in state],
                             [action])
        return active_tiles

    def value(self, state: List[int]):
        active_tiles = self._get_active_tiles(state)
        return np.sum(self._weights[active_tiles])

    def learn(self, past_state, next_state, reward):
        # apply differential sarsa
        active_tiles = self._get_active_tiles(past_state)
        delta = reward - self._average_reward + self.value(next_state) - self.value(past_state)
        self._average_reward += self._beta * delta
        delta *= self._alpha

        for active_tile in active_tiles:
            self._weights[active_tile] += delta

        return delta

    def stats(self):
        """Returns an array of internal stats"""
        return [self._average_reward]
    


if __name__ == "__main__":
    app = DSPSarsaPolynomial(state_dim=5, polynomial_order=1)
    print(app.value([1, 2, 3, 4, 5]))
    print(app.value_of_feature([1, 2, 3, 4, 5], "11111"))
