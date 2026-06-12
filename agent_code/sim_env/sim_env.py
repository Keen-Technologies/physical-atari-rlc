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

# sim_env.py
#
# Atari environment wrapper that encapsulates ALE-py logic

import numpy as np
from ale_py import ALEInterface, Action, LoggerMode, roms


class LatencyModel:
    """
    Wraps Atari joystick actions with a learned model to simulate real-world latency.
    Maintains a queue of past actions and uses a small neural network to determine
    which action should be executed, based on recent input history.
    """

    def __init__(self, path_to_model_weights=".."):
        """
        Initializes the latency model by loading neural network weights and
        initializing the action queue with NOOP actions.

        Args:
            path_to_model_weights (str): Path to the npz file containing the model's weight paramerters
        """
        self.action_queue = []
        for _ in range(30):  # Start with 30 NOOPs to fill the buffer
            self.action_queue.append(self.__one_hot_encode(0, 0, 36))
        weights = np.load(path_to_model_weights)
        self.fc_weight = weights["fc_weight"]
        self.fc_bias = weights["fc_bias"]
        self.pred_weight = weights["pred_weight"]
        self.pred_bias = weights["pred_bias"]
        self.last_action = 0

    def __one_hot_encode(self, value, value_2, length):
        """
        Returns a one-hot encoded vector.

        Args:
            value (int): Index to be set as 1.
            value_2 (int): Index in the second half to be set as 1
            length (int): Length of the output vector.

        Returns:
            list[float]: One-hot encoded vector.
        """
        vec = [0.0] * length
        vec[value] = 1.0
        vec[18 + value_2] = 1.0
        return vec

    def __forward(self, x):
        """
        Runs a forward pass through the two-layer MLP with ReLU activation.

        Args:
            x (np.ndarray): Input vector of shape (1, 30 * 18)

        Returns:
            np.ndarray: Output logits from the final layer.
        """
        x = x @ self.fc_weight.T + self.fc_bias
        x = np.maximum(x, 0)  # ReLU
        x = x @ self.pred_weight.T + self.pred_bias
        return x

    def act(self, action):
        """
        Accepts a new joystick action, updates the action queue, and returns
        the predicted action to execute (with latency effects).

        Args:
            action (int): The new joystick action.

        Returns:
            int: The action to actually execute, based on the model's prediction.
        """
        # model input is history of the past 30 actions
        if len(self.action_queue) == 30:
            self.action_queue.pop(0)
        self.action_queue.append(self.__one_hot_encode(action, self.last_action, 36))
        representation = np.array(self.action_queue).reshape(1, -1)
        logits = self.__forward(representation)
        probs = np.exp(logits - np.max(logits))  # for numerical stability
        probs /= np.sum(probs)
        sampled_action = int(np.argmax(probs[0]))
        self.last_action = sampled_action
        return sampled_action


class SimEnv:
    """
    Atari environment wrapper that provides a clean interface to ALE-py.
    Handles latency models and episode termination.
    """

    def __init__(
        self,
        game,
        seed=0,
        latency_model=None,
        max_frames_without_reward=18_000,
        fps=60,
        use_reduced_action_set=False
    ):
        """
        Initialize the Atari environment.

        Args:
            game (str): Name of the Atari game (e.g., "pong", "breakout")
            seed (int): Random seed for reproducibility
            latency_model (LatencyModel): Optional latency model to simulate physical delays
            max_frames_without_reward (int): Truncate episode after this many frames without reward
            fps (int): Target FPS for the environment (default: 60, ALE native speed)
                      FPS must be <= 60 and 60 must be divisible by fps
            use_reduced_action_set (bool): If True, use minimal action set for the game (default: False)
        """
        self.game = game
        self.seed = seed
        self.latency_model = latency_model
        self.max_frames_without_reward = max_frames_without_reward
        self.use_reduced_action_set = use_reduced_action_set
        
        # Validate and calculate action repeat based on FPS
        if fps > 60:
            raise ValueError(f"FPS cannot be greater than 60 (ALE native speed). Got: {fps}")
        if 60 % fps != 0:
            raise ValueError(f"FPS must evenly divide 60. Got: {fps}")
        
        self.fps = fps
        self.action_repeat = 60 // fps
        print(f"[SimEnv] FPS: {fps}, action_repeat: {self.action_repeat}")

        # Initialize ALE
        self.ale = ALEInterface()
        self.ale.setLoggerMode(LoggerMode.Error)
        self.ale.setInt('random_seed', seed)
        self.ale.setFloat('repeat_action_probability', 0.0)

        # Load the game
        game_path = roms.get_rom_path(game)
        self.ale.loadROM(game_path)

        # Use minimal or full legal action set based on parameter.
        # ale-py 0.12+ returns Action enums; store integer values for ALE.act().
        if use_reduced_action_set:
            self.action_set = [action.value for action in self.ale.getMinimalActionSet()]
            print(f"[SimEnv] Using minimal action set with {len(self.action_set)} actions")
        else:
            self.action_set = [action.value for action in self.ale.getLegalActionSet()]
            print(f"[SimEnv] Using legal action set with {len(self.action_set)} actions")

        # Initialize state
        self.previous_lives = self.ale.lives()
        self.frames_without_reward = 0
        
        # Current step results
        self._observation = None
        self._reward = 0
        self._terminated = False
        self._truncated = False
        if self.latency_model is not None:
            print("Using latency model")

    def step(self, action_index):
        """
        Execute one step in the environment.

        Args:
            action_index (int): Index into the action_set (not the raw ALE action)
        """
        # Convert action index to ALE action (action_set stores integer Action values)
        agent_action = self.action_set[action_index]
        
        # Apply latency model if present
        if self.latency_model is not None:
            taken_action = self.latency_model.act(agent_action)
        else:
            taken_action = agent_action
        
        # Execute action in ALE with action repeat
        self._reward = 0
        for _ in range(self.action_repeat):
            frame_reward = self.ale.act(taken_action)
            self._reward += frame_reward
            
            # Update frames without reward counter
            if frame_reward != 0:
                self.frames_without_reward = 0
            else:
                self.frames_without_reward += 1
            
            # Check for truncation (too many frames without reward)
            if self.frames_without_reward >= self.max_frames_without_reward:
                break
            
            # Check for game over (break early if episode ends)
            if self.ale.game_over():
                break
            
            # Check for life loss
            current_lives = self.ale.lives()
            if current_lives < self.previous_lives:
                break
        
        # Check termination conditions
        self._terminated = False
        self._truncated = False
        
        # Check for life loss
        current_lives = self.ale.lives()
        if current_lives < self.previous_lives:
            self.previous_lives = current_lives
            self._terminated = True
        
        # Check for game over
        if self.ale.game_over():
            self._terminated = True
        
        # Check for truncation (too many frames without reward)
        if self.frames_without_reward >= self.max_frames_without_reward:
            self._truncated = True
        
        # Get observation
        self._observation = self.ale.getScreenRGB()

    def perceive(self):
        pass 
    
    def get_observation(self):
        """Returns the current RGB observation (210x160x3)."""
        return self._observation

    def get_reward(self):
        """Returns the reward from the last step."""
        return self._reward

    def get_terminated(self):
        """Returns True if episode ended (life loss or game over)."""
        return self._terminated

    def get_truncated(self):
        """Returns True if episode was truncated (max frames without reward)."""
        return self._truncated

    def reset(self):
        """Reset the environment for a new episode."""
        self.ale.reset_game()
        self.previous_lives = self.ale.lives()
        self.frames_without_reward = 0
        self._reward = 0
        self._terminated = False
        self._truncated = False
        self._observation = self.ale.getScreenRGB()

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
        """Returns the current RAM state."""
        return self.ale.getRAM()

