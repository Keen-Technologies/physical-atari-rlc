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
import sys
import os
import struct
import random
import numpy as np
import json
import itertools
import sys
import os
# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(__file__))
from sim_env import SimEnv, LatencyModel
from checkpoint_store import (
    make_run_dir,
    load_checkpoint,
    save_checkpoint,
    apply_checkpoint_to_agent,
)
import torch


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
    benchmark_keys = {'game', 'mingame', 'physical', 'steps',
                      'train', 'seed', 'module_name', 'device', 'load_model', 'save_model',
                      'store_weights', 'exp_name', 'checkpoint_dir', 'checkpoint_load_path',
                      'max_frames_without_reward', 'latency_model', 'latency_model_path', 'env', 'eval_steps',
                      'fps', 'light_environment',
                      'use_reduced_action_set', 'load_weights',
                      'delay_learning_by_steps'}
    parms = {k: (typed_value(str(v)) if isinstance(v, str) else v)
             for k, v in agent_parms.items() if k not in benchmark_keys}



    # Create the environment
    print(f'loading {agent_parms["game"]}')
    
    env_type = agent_parms['env']
    
    if env_type == 'real':
        from real_env import RealEnv

        print("Using physical Atari environment")
        # Determine if latency model should be used (for RealEnv, latency_model is not used but kept for compatibility)
        latency_model_instance = None
        # Note: RealEnv doesn't use latency_model, but we pass it for interface compatibility
        
        env = RealEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=latency_model_instance,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            config_path=os.path.expanduser("~/robotroller.conf"),
            exp_name=run_name,
            use_reduced_action_set=agent_parms['use_reduced_action_set']
        )
        # Store robotroller.conf details
        config_path = os.path.expanduser("~/robotroller.conf")
        with open(config_path, "r") as f:
            results['robotroller_config'] = json.load(f)
        print(f"[learn_policy] Stored robotroller.conf configuration")
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
