#!/usr/bin/python
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

# real_env.py
#
# Physical Atari environment wrapper that mirrors sim_env.py interface
# Uses C++ PhysicalAtariEnv which integrates camera, AprilTags, and RoboTroller
#
# Physical runs require the user to launch PhysicalALE on the devbox manually
# before training (e.g. PhysicalALE ./games/ pong).

import json
import numpy as np
import os
from ale_py import ALEInterface, roms


class RealEnv:
    """
    Physical Atari environment wrapper that provides a clean interface matching sim_env.py.
    This is now a thin wrapper around the C++ PhysicalAtariEnv class.
    """

    def __init__(
        self,
        game,
        seed=0,
        latency_model=None,
        max_frames_without_reward=18_000,
        config_path=os.path.expanduser("~/robotroller.conf"),
        exp_name="",
        use_reduced_action_set=False
    ):
        """
        Initialize the physical Atari environment.

        Args:
            game (str): Name of the Atari game (e.g., "pong", "breakout")
            seed (int): Random seed for reproducibility
            latency_model: Optional latency model (not used in physical env, kept for interface compatibility)
            max_frames_without_reward (int): Truncate episode after this many frames without reward
            config_path (str): Path to robotroller.conf file with camera and robot settings (default: ~/robotroller.conf)
            exp_name (str): Experiment name (kept for interface compatibility)
            use_reduced_action_set (bool): If True, use minimal action set for the game (default: False)
        """
        self.game = game
        self.seed = seed
        self.latency_model = latency_model  # Not used in physical env, but kept for compatibility
        self.exp_name = exp_name
        self.use_reduced_action_set = use_reduced_action_set

        # Load configuration
        with open(config_path, "r") as f:
            config = json.load(f)

        # Extract camera configuration
        camera_device_str = config["camera"]["device"]
        # Extract camera index from device string (e.g., "/dev/video0" -> 0)
        if isinstance(camera_device_str, str) and "/dev/video" in camera_device_str:
            camera_device = int(camera_device_str.split("/dev/video")[-1])
        elif isinstance(camera_device_str, int):
            camera_device = camera_device_str
        else:
            camera_device = int(camera_device_str)
        
        camera_focus = config["camera"]["focus"]
        camera_zoom = config["camera"].get("zoom", 100)
        camera_width = config["camera"]["width"]
        camera_height = config["camera"]["height"]
        camera_fps = config["camera"].get("fps", 30)
        camera_exposure = config["camera"].get("exposure", 20)
        camera_brightness = config["camera"].get("brightness", 128)
        camera_contrast = config["camera"].get("contrast", 128)

        # Extract robot configuration
        serial_port = config["robot"]["serial_port"]
        position_d_gain = config["robot"]["D_gain"]
        position_i_gain = config["robot"]["I_gain"]
        position_p_gain = config["robot"]["P_gain"]
        baud_rate = config["robot"]["baud_rate"]
        
        # Extract servo position configuration
        dpad_servo_default = config["robot"]["dpad_servo_default"]
        dpad_servo_right = config["robot"]["dpad_servo_right"]
        dpad_servo_left = config["robot"]["dpad_servo_left"]
        dpad_servo_up = config["robot"]["dpad_servo_up"]
        dpad_servo_down = config["robot"]["dpad_servo_down"]
        button_servo_default = config["robot"]["button_servo_default"]
        button_deflection = config["robot"]["button_deflection"]

        # Initialize ALE to get minimal action set if needed
        if use_reduced_action_set:
            print(f"[RealEnv] Initializing ALE to get minimal action set for {game}")
            temp_ale = ALEInterface()
            game_path = roms.get_rom_path(game)
            temp_ale.loadROM(game_path)
            minimal_action_set = temp_ale.getMinimalActionSet()
            # Convert Action enums to ints
            self.action_set = [action.value for action in minimal_action_set]
            print(f"[RealEnv] Using minimal action set with {len(self.action_set)} actions: {self.action_set}")
        else:
            # Use full action set: all 18 actions (NOOP through DOWNLEFTFIRE)
            self.action_set = list(range(18))
            print(f"[RealEnv] Using legal action set with {len(self.action_set)} actions")
        
        # Initialize C++ PhysicalAtariEnv (does all the heavy lifting)
        import robotroller

        print(f"[RealEnv] Initializing C++ PhysicalAtariEnv for game: {game}")
        self.env = robotroller.PhysicalAtariEnv(
            game,
            seed,
            camera_device,
            camera_width,
            camera_height,
            camera_focus,
            camera_zoom,
            camera_fps,
            camera_exposure,
            camera_brightness,
            camera_contrast,
            serial_port,
            position_d_gain,
            position_i_gain,
            position_p_gain,
            baud_rate,
            dpad_servo_default,
            dpad_servo_right,
            dpad_servo_left,
            dpad_servo_up,
            dpad_servo_down,
            button_servo_default,
            button_deflection
        )
        
        # Cache for last step results
        self._observation = None
        self._reward = 0
        self._terminated = False
        self._truncated = False
        
        print(f"[RealEnv] Environment initialized successfully")

    def step(self, action_index):
        """
        Execute one step in the environment.

        Args:
            action_index (int): Index into the action_set
        """
        # Map action index to the actual action from action_set
        actual_action = self.action_set[action_index]
        # Delegate to C++ implementation
        self.env.step(actual_action)
    

    def perceive(self):
        """Perceive the current state of the environment."""
        self._observation = self.env.getObservation()
        self._reward = self.env.getReward()
        self._terminated = self.env.getTerminated()
        self._truncated = self.env.getTruncated()

    def get_observation(self):
        """Returns the current RGB observation (210x160x3)."""
        return self._observation

    def get_reward(self):
        """Returns the reward from the last step (score change)."""
        return self._reward

    def get_terminated(self):
        """Returns True if episode ended (game over)."""
        return self._terminated

    def get_truncated(self):
        """Returns True if episode was truncated (max frames without reward)."""
        return self._truncated

    def reset(self):
        """Reset the environment for a new episode."""
        self.env.reset()
        self._observation = self.env.getObservation()
        self._reward = 0
        self._terminated = False
        self._truncated = False

    def get_num_actions(self):
        """Returns the number of valid actions in the action space."""
        return len(self.action_set)

    def get_number_of_actions(self):
        """Returns the number of actions in the action space.
        
        Returns the size of the minimal action set if use_reduced_action_set was True
        in the constructor, otherwise returns the size of the legal action set.
        """
        return len(self.action_set)

    def get_ram(self):
        """Returns the current RAM state (not available for physical Atari)."""
        # RAM is not accessible from physical hardware
        return np.zeros(128, dtype=np.uint8)

    def shutdown(self):
        """Clean up resources."""
        pass

