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

# learn_policy.py 6/9/2025
#
# Harness to run benchmarks on an atari agent
#
# learn_policy.py job_number agent_module [--parm_key parm_value]
import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import argparse
import importlib
import time
from datetime import timedelta, datetime
import struct
import random
import numpy as np
import json
import itertools
import io
import pickle
# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(__file__))
from sim_env import SimEnv, LatencyModel
import torch

WEIGHTS_FILENAME = "weights.pkl"
LAST_OBSERVATION_FILENAME = "last_observation.pkl"
AGENT_STATE_FILENAME = "agent_state.pkl"
RESULTS_FILENAME = "results.json"

BENCHMARK_KEYS = {
    'game', 'mingame', 'physical', 'steps',
    'train', 'seed', 'module_name', 'device', 'load_model', 'save_model',
    'store_weights', 'exp_name', 'checkpoint_dir', 'checkpoint_load_path',
    'max_frames_without_reward', 'latency_model', 'latency_model_path', 'env', 'eval_steps',
    'fps', 'light_environment', 'use_reduced_action_set',
    'load_weights', 'delay_learning_by_steps', 'robotroller_config_path',
}


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


def apply_checkpoint_to_agent(agent, run_dir, device, mode='eval'):
    """
    Load weights and observation state from a checkpoint directory into agent.

    mode: 'eval' or 'train' restores ring/buffer for continued eval/training;
          'finetune' sets f=0 so the ring fills naturally during fine-tuning.
    """
    weights_path = os.path.join(run_dir, WEIGHTS_FILENAME)
    if os.path.isfile(weights_path):
        print(f"Loading model weights from {weights_path}")
        with open(weights_path, 'rb') as f:
            class DeviceUnpickler(pickle.Unpickler):
                def __init__(self, file, target_device):
                    super().__init__(file)
                    self.target_device = target_device

                def find_class(self, module, name):
                    if module == 'torch.storage' and name == '_load_from_bytes':
                        return lambda b: torch.load(io.BytesIO(b), map_location=self.target_device)
                    return super().find_class(module, name)

            try:
                weights_dict = DeviceUnpickler(f, device).load()
            except Exception as e:
                print(f"Warning: Custom unpickler failed ({e}), trying standard pickle.loads")
                f.seek(0)
                weights_dict = pickle.loads(f.read())
        if isinstance(weights_dict, torch.Tensor):
            weights_dict = weights_dict.to(device)
        elif isinstance(weights_dict, dict):
            queue = [weights_dict]
            while queue:
                d = queue.pop()
                for key, val in d.items():
                    if isinstance(val, torch.Tensor):
                        d[key] = val.to(device)
                    elif isinstance(val, dict):
                        queue.append(val)
                    elif isinstance(val, (list, tuple)):
                        d[key] = type(val)(
                            x.to(device) if isinstance(x, torch.Tensor) else x for x in val
                        )
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
    with open(obs_path, 'rb') as f:
        class DeviceUnpickler(pickle.Unpickler):
            def __init__(self, file, target_device):
                super().__init__(file)
                self.target_device = target_device

            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location=self.target_device)
                return super().find_class(module, name)

        try:
            last_obs = DeviceUnpickler(f, device).load()
        except Exception as e:
            print(f"Warning: Custom unpickler failed ({e}), trying standard pickle.loads")
            f.seek(0)
            last_obs = pickle.loads(f.read())
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
            with open(state_path, 'rb') as f:
                class DeviceUnpickler(pickle.Unpickler):
                    def __init__(self, file, target_device):
                        super().__init__(file)
                        self.target_device = target_device

                    def find_class(self, module, name):
                        if module == 'torch.storage' and name == '_load_from_bytes':
                            return lambda b: torch.load(io.BytesIO(b), map_location=self.target_device)
                        return super().find_class(module, name)

                try:
                    agent_state = DeviceUnpickler(f, device).load()
                except Exception as e:
                    print(f"Warning: Custom unpickler failed ({e}), trying standard pickle.loads")
                    f.seek(0)
                    agent_state = pickle.loads(f.read())
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


class Experiment:
    def __init__(self, json_path):
        with open(json_path, "r") as f:
            self.params = json.load(f)

        # Separate keys and value lists
        keys = list(self.params.keys())
        values = [v if isinstance(v, list) else [v] for v in self.params.values()]

        # Compute cross product of hyper-parameters
        self.configs = []
        for combo in itertools.product(*values):
            self.configs.append(dict(zip(keys, combo)))

        print("Total experiment configurations:", len(self.configs))

    def get_params(self, idx):
        return self.configs[idx]


def typed_value(s):
    try:
        int_val = int(s)
        # If int() succeeds, check that float would not change it
        if '.' in s or 'e' in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        try:
            float_val = float(s)
            return float_val
        except ValueError:
            return s


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--run", type=int, default=1, help="Run number (int)")
    parser.add_argument("--gpu", type=str, help="GPU device ID to use (sets CUDA_VISIBLE_DEVICES)")
    args = parser.parse_args()

    if not args.config:
        print("--config parameter must be passed")
        exit(1)

    my_exp = Experiment(args.config)
    agent_parms = my_exp.get_params(args.run)
    # Results dictionary with hyperparameters and results
    # Standardized naming convention
    results = {
        **agent_parms, 
        "training_avg_reward_history": [],         # List of (avg_reward, frame)
        "training_moving_avg_reward_history": [],  # List of (moving_avg, frame)
        "running_avg_reward_over_time": []         # List of running average (decayed by 0.999) every 500 steps
    }

    # Generate run name with timestamp
    date = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{agent_parms['game']}__{agent_parms['module_name']}__{agent_parms['seed']}__{date}"
    results['run_name'] = run_name

    start_time = time.time()
    np.random.seed(agent_parms['seed'])

    # Add remaining parameters to agent parms dict (excluding benchmark-level ones)
    parms = {k: (typed_value(str(v)) if isinstance(v, str) else v)
             for k, v in agent_parms.items() if k not in BENCHMARK_KEYS}



    # Create the environment
    print(f'loading {agent_parms["game"]}')
    
    env_type = agent_parms['env']
    
    if env_type == 'real':
        from real_env import RealEnv

        print("Using physical Atari environment")
        # Determine if latency model should be used (for RealEnv, latency_model is not used but kept for compatibility)
        latency_model_instance = None
        # Note: RealEnv doesn't use latency_model, but we pass it for interface compatibility
        
        config_path = agent_parms["robotroller_config_path"]

        env = RealEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=latency_model_instance,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            config_path=config_path,
            exp_name=run_name,
            use_reduced_action_set=agent_parms['use_reduced_action_set']
        )
        with open(config_path, "r") as f:
            results['robotroller_config'] = json.load(f)
        results['robotroller_config_path'] = config_path
        print(f"[learn_policy] Using robotroller config at {config_path}")
    elif env_type == 'sim':
        # Use simulated environment
        print("Using simulated Atari environment")
        # Determine if latency model should be used
        latency_model_instance = None
        if agent_parms['latency_model']:
            latency_model_instance = LatencyModel(agent_parms['latency_model_path'])
        env = SimEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=latency_model_instance,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            fps=agent_parms['fps'],
            use_reduced_action_set=agent_parms['use_reduced_action_set']
        )
    else:
        print(f"Error: unknown env type '{env_type}' (expected 'sim' or 'real')")
        sys.exit(1)

    num_actions = env.get_num_actions()
    # create the agent to test, passing hyperparameters from config
    _agent_mod = importlib.import_module(agent_parms['module_name'])
    agent = _agent_mod.Agent(
        agent_parms['seed'],
        num_actions,
        agent_parms['steps'],
        agent_parms['device'],
        **parms
    )
    # Load weights from checkpoint if load_weights is enabled
    if agent_parms.get('load_weights'):
        print("\n" + "=" * 50)
        print("Loading weights from checkpoint...")
        print("=" * 50)

        checkpoint_load_path = agent_parms.get('checkpoint_load_path')
        if not checkpoint_load_path:
            print("Error: load_weights is true but checkpoint_load_path is missing!")
            sys.exit(1)

        load_checkpoint(checkpoint_load_path)
        apply_checkpoint_to_agent(agent, checkpoint_load_path, agent_parms['device'], mode='train')
        print("Weight loading complete!")

    episode_scores = []
    running_episode_score = 0

    # reporting the frame rate per episode is useful
    episode_start_frame = 0
    episode_start_frame_time = time.time()

    agent_action_index = 0  # until a policy can be evaluated on observations

    # Track rewards for checkpoint logging
    cumulative_reward = 0
    moving_avg_reward = 0
    log_interval = 100  # Log every 500 frames


    # Initialize game state
    total_game_rewards = 0
    game_start_frame = 0
    env.reset()
    # Track timing for periodic prints
    training_start_time = time.time()
    last_print_time = training_start_time
    print_interval = 2.0  # Print every 2 seconds
    fps_frame_count = 0  # Track frames for FPS calculation

    # Run training loop with KeyboardInterrupt handling
    try:
        for frame in range(agent_parms['steps']):
            env.perceive()
            observation = env.get_observation()
            reward = env.get_reward()
            terminated = env.get_terminated()
            truncated = env.get_truncated()

            running_episode_score += reward
            cumulative_reward += reward
            moving_avg_reward += reward
            
            # Track FPS
            fps_frame_count += 1
            current_time = time.time()
            if current_time - last_print_time >= print_interval:
                elapsed = current_time - last_print_time
                fps = fps_frame_count / elapsed
                print(f"[learn_policy TRAIN] FPS: {fps:.2f}")
                print(f"[learn_policy TRAIN] frame: {frame}")
                fps_frame_count = 0
                last_print_time = current_time

            # Log metrics periodically
            if frame > 0 and frame % log_interval == 0:
                avg_reward = cumulative_reward / frame
                results["training_avg_reward_history"].append((avg_reward, frame))
                results["training_moving_avg_reward_history"].append((moving_avg_reward/log_interval, frame))
                moving_avg_reward = 0
                

            # Determine end of episode type for logging
            end_of_episode = 0
            if terminated:
                end_of_episode = 2  # game over or life loss
            elif truncated:
                end_of_episode = 3  # terminated without game over
                print(f'terminated at {agent_parms["max_frames_without_reward"]} frames without reward')

            if end_of_episode > 1:
                if frame > 0:
                    episode_scores.append(running_episode_score)
                    running_episode_score = 0
            
                env.reset()
                episode_start_frame = frame

            agent_action_index = agent.get_action(observation, reward, end_of_episode, train=True)
            env.step(agent_action_index)
            agent.learn(observation, reward, end_of_episode, train=True)
    except KeyboardInterrupt:
        print("\n[learn_policy] Received KeyboardInterrupt during training, shutting down...")
        # Cleanup environment before exiting - this disables motor torques for RealEnv
        if hasattr(env, 'shutdown'):
            env.shutdown()
        raise  # Re-raise to exit

    # ===== Evaluation Phase =====
    eval_steps = agent_parms['eval_steps']
    print("\n" + "=" * 50)
    print(f"Starting evaluation phase ({eval_steps} steps, train=False)")
    print("=" * 50)
    eval_episode_scores = []
    eval_running_episode_score = 0
    eval_cumulative_reward = 0
    eval_episode_start_frame = 0
    eval_episode_start_frame_time = time.time()

    # Reset environment for evaluation
    env.reset()
    agent_action_index = 0

    # Track timing for periodic prints in evaluation phase
    eval_start_time = time.time()
    eval_last_print_time = eval_start_time
    eval_agent_frame_timings = []  # Track agent.frame() call times in evaluation
    eval_fps_frame_count = 0  # Track frames for FPS calculation

    try:
        for eval_frame in range(eval_steps):
            # Step the environment with the agent's chosen action
            env.perceive()
    

            # Get the results
            observation = env.get_observation()
            reward = env.get_reward()
            terminated = env.get_terminated()
            truncated = env.get_truncated()


            # Track FPS and print periodically
            eval_fps_frame_count += 1
            eval_current_time = time.time()
            if eval_current_time - eval_last_print_time >= print_interval:
                elapsed = eval_current_time - eval_last_print_time
                fps = eval_fps_frame_count / elapsed
                print(f"[learn_policy EVAL] FPS: {fps:.2f}")
                eval_fps_frame_count = 0
                eval_last_print_time = eval_current_time

            eval_running_episode_score += reward
            eval_cumulative_reward += reward

            # Determine end of episode type for logging
            end_of_episode = 0
            if terminated:
                end_of_episode = 2  # game over or life loss
            elif truncated:
                end_of_episode = 3  # terminated without game over
                print(f'evaluation: terminated at {agent_parms["max_frames_without_reward"]} frames without reward')

            if end_of_episode > 1:
                if eval_frame > 0:
                    eval_episode_scores.append(eval_running_episode_score)
                    eval_running_episode_score = 0

                env.reset()
                eval_episode_start_frame = eval_frame

            # let the agent process the observation and choose next action (train=False for eval)
            eval_agent_frame_start = time.time()
            agent_action_index = agent.get_action(observation, reward, end_of_episode, train=False)
            env.step(agent_action_index)
            agent.learn(observation, reward, end_of_episode, train=False)
            eval_agent_frame_timings.append(time.time() - eval_agent_frame_start)
    except KeyboardInterrupt:
        print("\n[learn_policy] Received KeyboardInterrupt during evaluation, shutting down...")
        # Cleanup environment before exiting - this disables motor torques for RealEnv
        if hasattr(env, 'shutdown'):
            env.shutdown()
        raise  # Re-raise to exit

    # Store evaluation results
    results["eval_avg_reward_per_step"] = eval_cumulative_reward / eval_steps if eval_steps > 0 else 0
    results["eval_episode_scores"] = eval_episode_scores
    results["eval_total_episodes"] = len(eval_episode_scores)
    results["eval_mean_episode_score"] = np.mean(eval_episode_scores) if len(eval_episode_scores) > 0 else 0

    print(f"\nEvaluation complete:")
    print(f"  Average reward per step: {results['eval_avg_reward_per_step']:.4f}")
    print(f"  Total episodes: {len(eval_episode_scores)}")
    if len(eval_episode_scores) > 0:
        print(f"  Mean episode score: {np.mean(eval_episode_scores):.2f}")
        print(f"  Std episode score: {np.std(eval_episode_scores):.2f}")
    print("=" * 50 + "\n")

    if 'save_model' in agent_parms:
        filename = f'{agent_parms["save_model"]}/final.model'
        print('writing ' + filename)
        agent.save_model(filename)

    end_time = time.time()
    elapsed = timedelta(seconds=end_time - start_time)
    print(f'completed in {elapsed}')

    # Store final average reward
    if agent_parms['steps'] > 0:
        results["training_lifetime_avg_reward_per_step"] = cumulative_reward / agent_parms['steps']

    # Store episode scores
    results["training_episode_scores"] = episode_scores
    results["training_total_episodes"] = len(episode_scores)
    results["duration_seconds"] = end_time - start_time

    checkpoint_dir = agent_parms.get('checkpoint_dir', './checkpoints')
    run_dir = make_run_dir(checkpoint_dir, run_name)
    save_checkpoint(run_dir, results, agent, store_weights=agent_parms.get('store_weights', False))
    
    # Cleanup environment if it has a shutdown method
    if hasattr(env, 'shutdown'):
        env.shutdown()


if __name__ == '__main__':
    main(sys.argv)
