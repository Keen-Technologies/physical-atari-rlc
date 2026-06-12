# Copyright 2026 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random


class Agent:
    """Agent that picks a uniform random action on every frame. For testing the stack."""

    def __init__(self, seed, num_actions, total_frames, device='cuda', **kwargs):
        random.seed(seed)
        self.num_actions = num_actions
        self.device = device
        self.total_frames = total_frames
        self.f = 0

    def get_action(self, observation_rgb8, reward, end_of_episode, train=True):
        self.f += 1
        return random.randint(0, self.num_actions - 1)

    def learn(self, observation_rgb8, reward, end_of_episode, train=True, update_weights=True):
        pass

    def get_state(self):
        return {}

    def load_state(self, state_dict):
        pass
