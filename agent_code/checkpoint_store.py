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

import io
import json
import os
import pickle
import sys

import torch

WEIGHTS_FILENAME = "weights.pkl"
LAST_OBSERVATION_FILENAME = "last_observation.pkl"
AGENT_STATE_FILENAME = "agent_state.pkl"
RESULTS_FILENAME = "results.json"

RESULTS_KEYS_TO_STRIP = [
    'weights_file_id', 'last_observation_file_id', 'agent_state_file_id', 'run_name',
    'avg_reward_over_time', 'moving_avg_reward_over_time', 'episode_scores',
    'episodic_score_eval', 'elapsed_time_seconds', 'total_episodes',
    'lifetime_avg_reward', 'avg_reward_eval', 'total_eval_episodes',
    'mean_episodic_score_eval',
    'training_avg_reward_history', 'training_moving_avg_reward_history',
    'training_episode_scores', 'training_total_episodes', 'duration_seconds',
    'training_lifetime_avg_reward_per_step', 'eval_avg_reward_per_step',
    'eval_episode_scores', 'eval_total_episodes', 'eval_mean_episode_score',
]

BENCHMARK_KEYS = {
    'game', 'mingame', 'physical', 'steps',
    'train', 'seed', 'module_name', 'device', 'load_model', 'save_model',
    'store_weights', 'exp_name', 'checkpoint_dir', 'checkpoint_load_path',
    'max_frames_without_reward', 'latency_model', 'latency_model_path', 'env', 'eval_steps',
    'fps', 'light_environment', 'use_reduced_action_set',
    'load_weights', 'delay_learning_by_steps',
}


def _move_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_move_to_device(item, device) for item in obj)
    return obj


def _load_pickle_with_device(path, device):
    class DeviceUnpickler(pickle.Unpickler):
        def __init__(self, file, target_device):
            super().__init__(file)
            self.target_device = target_device

        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(io.BytesIO(b), map_location=self.target_device)
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        try:
            obj = DeviceUnpickler(f, device).load()
        except Exception as e:
            print(f"Warning: Custom unpickler failed ({e}), trying standard pickle.loads")
            f.seek(0)
            obj = pickle.loads(f.read())
    return _move_to_device(obj, device)


def make_run_dir(checkpoint_dir, run_name):
    run_dir = os.path.join(checkpoint_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def load_checkpoint(run_dir):
    """Load results.json from a checkpoint run directory."""
    results_path = os.path.join(run_dir, RESULTS_FILENAME)
    if not os.path.isfile(results_path):
        print(f"Error: checkpoint not found at {results_path}")
        sys.exit(1)

    with open(results_path, 'r') as f:
        model_doc = json.load(f)

    print(f"Loaded checkpoint: {run_dir}")
    print(f"Run name: {model_doc.get('run_name', 'Unknown')}")
    return model_doc


def extract_agent_parms(model_doc, extra_benchmark_keys=None):
    agent_parms = model_doc.copy()
    for key in RESULTS_KEYS_TO_STRIP:
        agent_parms.pop(key, None)

    benchmark_keys = set(BENCHMARK_KEYS)
    if extra_benchmark_keys:
        benchmark_keys.update(extra_benchmark_keys)

    return agent_parms, benchmark_keys


def save_checkpoint(run_dir, results, agent, store_weights=True):
    """Save results.json and optional weight/observation artifacts to run_dir."""
    os.makedirs(run_dir, exist_ok=True)

    if store_weights:
        weights_path = os.path.join(run_dir, WEIGHTS_FILENAME)
        with open(weights_path, 'wb') as f:
            pickle.dump(agent.get_state(), f)
        results['weights_path'] = WEIGHTS_FILENAME
        print(f"Model weights saved to {weights_path}")

        last_observation = None
        buffer_type = None
        last_obs_index = None

        if hasattr(agent, 'observation_buffer'):
            last_obs_index = (agent.f - 1) % agent.total_frames
            last_observation = agent.observation_buffer[last_obs_index].clone().cpu().numpy()
            buffer_type = "observation_buffer"
        elif hasattr(agent, 'observation_ring'):
            last_obs_index = (agent.f - 1) % agent.ring_size
            last_observation = agent.observation_ring[last_obs_index].clone().cpu().numpy()
            buffer_type = "observation_ring"

            if hasattr(agent, 'observation_ema') and hasattr(agent, 'action_ema'):
                agent_state = {
                    'observation_ema': agent.observation_ema.clone().cpu().numpy(),
                    'action_ema': agent.action_ema.clone().cpu().numpy(),
                    'selected_action_index': agent.selected_action_index,
                    'f': agent.f,
                }
                state_path = os.path.join(run_dir, AGENT_STATE_FILENAME)
                with open(state_path, 'wb') as f:
                    pickle.dump(agent_state, f)
                results['agent_state_path'] = AGENT_STATE_FILENAME
                print(f"Agent state saved to {state_path}")
        elif hasattr(agent, 'observation_ema'):
            last_observation = agent.observation_ema.clone().cpu().numpy()
            last_obs_index = agent.f - 1
            buffer_type = "observation_ema"
        else:
            print("Warning: Agent has no observation buffer/ring/ema, skipping last observation storage")

        if last_observation is not None:
            obs_path = os.path.join(run_dir, LAST_OBSERVATION_FILENAME)
            with open(obs_path, 'wb') as f:
                pickle.dump(last_observation, f)
            results['last_observation_path'] = LAST_OBSERVATION_FILENAME
            results['last_observation_buffer_type'] = buffer_type
            results['last_observation_index'] = last_obs_index
            print(f"Last observation ({buffer_type}) saved to {obs_path} (index {last_obs_index})")

    results_path = os.path.join(run_dir, RESULTS_FILENAME)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {results_path}")
    return run_dir


def save_results_json(path, results):
    dirpath = os.path.dirname(path)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {path}")


def apply_checkpoint_to_agent(agent, run_dir, device, mode='eval'):
    """
    Load weights and observation state from a checkpoint directory into agent.

    mode: 'eval' or 'train' restores ring/buffer for continued eval/training;
          'finetune' sets f=0 so the ring fills naturally during fine-tuning.
    """
    weights_path = os.path.join(run_dir, WEIGHTS_FILENAME)
    if os.path.isfile(weights_path):
        print(f"Loading model weights from {weights_path}")
        weights_dict = _load_pickle_with_device(weights_path, device)
        if hasattr(agent, 'load_state'):
            agent.load_state(weights_dict)
            print("Weights loaded successfully into agent (both model and target_model)")
            if hasattr(agent, 'trained'):
                agent.trained = True
                print("Agent marked as trained")
        else:
            print("Warning: Agent does not have load_state() method")
    else:
        print("Warning: No weights file found in checkpoint!")

    obs_path = os.path.join(run_dir, LAST_OBSERVATION_FILENAME)
    if not os.path.isfile(obs_path):
        print("Error: last_observation.pkl not found in checkpoint!")
        sys.exit(1)

    print(f"Loading last observation from {obs_path}")
    last_obs = _load_pickle_with_device(obs_path, device)
    if not isinstance(last_obs, torch.Tensor):
        last_obs_tensor = torch.from_numpy(last_obs).to(agent.device)
    else:
        last_obs_tensor = last_obs.to(agent.device)

    state_path = os.path.join(run_dir, AGENT_STATE_FILENAME)

    if hasattr(agent, 'observation_buffer'):
        agent.observation_buffer[0] = last_obs_tensor
        agent.f = 1
        print("Restored last observation buffer value and set f=1")
    elif hasattr(agent, 'observation_ring'):
        if os.path.isfile(state_path):
            print(f"Loading agent state from {state_path}")
            agent_state = _load_pickle_with_device(state_path, device)
            if not isinstance(agent_state, dict):
                agent_state = agent_state

            if 'observation_ema' in agent_state and hasattr(agent, 'observation_ema'):
                ema = agent_state['observation_ema']
                agent.observation_ema = torch.from_numpy(ema).to(agent.device) if not isinstance(ema, torch.Tensor) else ema.to(agent.device)
                print("Restored observation_ema")

            if 'action_ema' in agent_state and hasattr(agent, 'action_ema'):
                ema = agent_state['action_ema']
                agent.action_ema = torch.from_numpy(ema).to(agent.device) if not isinstance(ema, torch.Tensor) else ema.to(agent.device)
                print("Restored action_ema")

            if 'selected_action_index' in agent_state:
                agent.selected_action_index = agent_state['selected_action_index']
                print(f"Restored selected_action_index to {agent.selected_action_index}")

            stored_f = agent_state.get('f', 'not stored')
            if mode == 'finetune':
                agent.f = 0
                print(f"Set f to 0 for fine-tuning (original stored f: {stored_f})")
                print("Ring buffer will fill naturally as frames progress")
            else:
                agent.f = agent.ring_size
                agent.observation_ring[0] = agent.observation_ema.to(dtype=torch.uint8)
                agent.action_ema_ring[0] = agent.action_ema
                print(f"Restored f to {agent.f} (from stored value {stored_f})")
                print("Populated observation_ring[0] and action_ema_ring[0] with restored values")
        else:
            print("Warning: agent_state.pkl not found - observation_ema and action_ema will start at zeros")
            if mode == 'finetune':
                agent.f = 0
                print("Set f to 0 (fallback - no agent state found)")
            else:
                agent.observation_ring[0] = last_obs_tensor
                agent.f = agent.ring_size
                print(f"Restored last observation ring value to ring[0] and set f={agent.f} (fallback)")
    elif hasattr(agent, 'observation_ema'):
        agent.observation_ema = last_obs_tensor
        agent.f = 1
        print("Restored last observation ema value and set f=1")
    else:
        print("Error: Agent does not have observation_buffer, observation_ring, or observation_ema attribute!")
        sys.exit(1)

    print("Checkpoint applied to agent")
