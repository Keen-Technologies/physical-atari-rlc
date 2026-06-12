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

# evaluate_policy.py
#
# Script to load a trained model from a checkpoint directory and evaluate it
#
# Usage: python evaluate_policy.py --checkpoint-dir ./checkpoints/pong__...__0__<timestamp>

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import importlib
import time
import json
import numpy as np
import copy
from datetime import datetime, timedelta
import sys
import torch

# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(__file__))
from sim_env import SimEnv, LatencyModel
from checkpoint_store import (
    load_checkpoint,
    extract_agent_parms,
    apply_checkpoint_to_agent,
    save_results_json,
)


def typed_value(s):
    """Convert string to appropriate type (int, float, or str)"""
    try:
        int_val = int(s)
        if '.' in s or 'e' in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        try:
            float_val = float(s)
            return float_val
        except ValueError:
            return s


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained model from a checkpoint directory')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Path to checkpoint run directory')
    parser.add_argument('--eval_steps', type=int, default=20000, help='Number of evaluation steps')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use for evaluation (e.g., cpu, cuda, mps)')
    parser.add_argument('--env', type=str, default='sim', help='Environment type (sim or real)')
    parser.add_argument('--comment', type=str, default='no comment', help='Optional comment to log with evaluation results')
    args = parser.parse_args()

    model_doc = load_checkpoint(args.checkpoint_dir)
    print("Training hyperparameters:")
    for key in ['game', 'module_name', 'seed', 'steps']:
        if key in model_doc:
            print(f"  {key}: {model_doc[key]}")

    agent_parms, benchmark_keys = extract_agent_parms(model_doc)
    
    parms = {k: (typed_value(str(v)) if isinstance(v, str) else v)
             for k, v in agent_parms.items() if k not in benchmark_keys}

    # Generate unique run name for evaluation
    eval_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_run_name = model_doc['run_name']
    eval_run_name = f"{source_run_name}__eval__{eval_timestamp}"
    print(f"\nEvaluation run name: {eval_run_name}")

    # Create the environment
    print(f"Creating environment: {agent_parms['game']}")
    
    # Variable to store robotroller config if using real environment
    robotroller_config = None
    
    if args.env == 'real':
        from real_env import RealEnv

        print("Using physical Atari environment")
        env = RealEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=None,  # RealEnv doesn't use latency_model
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            config_path=os.path.expanduser("~/robotroller.conf"),
            exp_name=eval_run_name,
            use_reduced_action_set=agent_parms['use_reduced_action_set']
        )
        
        # Store robotroller.conf details
        config_path = os.path.expanduser("~/robotroller.conf")
        with open(config_path, "r") as f:
            robotroller_config = json.load(f)
        print(f"[evaluate_policy] Stored robotroller.conf configuration")
    elif args.env == 'sim':
        # Use simulated environment
        print("Using simulated Atari environment")
        # Correctly check the latency_model boolean flag (matching learn_policy.py)
        latency_model_instance = None
        if agent_parms.get('latency_model'):
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
        print(f"Error: unknown env type '{args.env}' (expected 'sim' or 'real')")
        sys.exit(1)

    num_actions = env.get_num_actions()

    # Create the agent
    print(f"Creating agent: {agent_parms['module_name']}")
    print(f"Using device: {args.device}")
    agent = importlib.import_module(agent_parms['module_name']).Agent(
        agent_parms['seed'],
        num_actions,
        args.eval_steps,  # Use eval steps instead of training steps
        args.device,
        ring_size=10000,  # Use smaller ring buffer to save memory during evaluation
        **parms
    )

    apply_checkpoint_to_agent(agent, args.checkpoint_dir, args.device, mode='eval')

    # ===== Evaluation Phase =====
    print("\n" + "=" * 50)
    print(f"Starting evaluation phase ({args.eval_steps} steps, train=False)")
    print("=" * 50)

    eval_steps = args.eval_steps
    eval_episode_scores = []
    eval_running_episode_score = 0
    eval_cumulative_reward = 0
    eval_episode_start_frame = 0
    eval_episode_start_frame_time = time.time()

    # Reset environment for evaluation (but keep agent state with filled buffers)
    env.reset()
    agent_action_index = 0

    for eval_frame in range(eval_steps):
        # Step the environment with the agent's chosen action
        env.perceive()

        # Get the results
        observation = env.get_observation()
        reward = env.get_reward()
        terminated = env.get_terminated()
        truncated = env.get_truncated()

        eval_running_episode_score += reward
        eval_cumulative_reward += reward

        # Print current episodic score every 4000 steps
        if eval_frame > 0 and eval_frame % 4000 == 0:
            print(f'Step {eval_frame}: Current episodic score = {eval_running_episode_score:.2f}')

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

                # calculate step speed
                now = time.time()
                frames = eval_frame - eval_episode_start_frame
                frames_per_second = frames / (now - eval_episode_start_frame_time)
                eval_episode_start_frame_time = now

                # average over the last ten episodes
                total = 0
                count = 0
                for i in range(min(10, len(eval_episode_scores))):
                    total += eval_episode_scores[-1 - i]
                    count += 1
                avg = total / count if count > 0 else 0

                print(
                    f'EVAL {agent_parms["game"]} frame:{eval_frame:7} {frames_per_second:4.0f}/s ep: {len(eval_episode_scores) - 1:3},{frames:5}={int(eval_episode_scores[-1]):5} ep_avg:{avg:7.1f}')

            env.reset()
            eval_episode_start_frame = eval_frame

        # let the agent process the observation and choose next action (train=False for eval)
        agent_action_index = agent.get_action(observation, reward, end_of_episode, train=False)
        env.step(agent_action_index)
        agent.learn(observation, reward, end_of_episode, train=False)

    # Store evaluation results
    eval_avg_reward_per_step = eval_cumulative_reward / eval_steps if eval_steps > 0 else 0
    eval_mean_episode_score = np.mean(eval_episode_scores) if len(eval_episode_scores) > 0 else 0

    print(f"\nEvaluation complete:")
    print(f"  Average reward per step: {eval_avg_reward_per_step:.4f}")
    print(f"  Total episodes: {len(eval_episode_scores)}")
    if len(eval_episode_scores) > 0:
        print(f"  Mean episode score: {eval_mean_episode_score:.2f}")
        print(f"  Std episode score: {np.std(eval_episode_scores):.2f}")
    print("=" * 50 + "\n")

    # Prepare results document for saving
    eval_results = {
        'eval_run_name': eval_run_name,
        'eval_timestamp': eval_timestamp,
        'source_checkpoint_dir': args.checkpoint_dir,
        'source_run_name': model_doc['run_name'],
        'eval_steps': eval_steps,
        'eval_avg_reward_per_step': eval_avg_reward_per_step,
        'eval_episode_scores': eval_episode_scores,
        'eval_total_episodes': len(eval_episode_scores),
        'eval_mean_episode_score': eval_mean_episode_score,
        'comment': args.comment,
    }

    # Include robotroller config if using real environment
    if robotroller_config is not None:
        eval_results['robotroller_config'] = robotroller_config

    # Include all original hyperparameters
    # Use the filtered agent_parms which has results stripped out
    eval_results['hyperparameters'] = agent_parms

    eval_results_path = os.path.join(args.checkpoint_dir, 'eval_results.json')
    save_results_json(eval_results_path, eval_results)
    
    # Cleanup environment if it has a shutdown method
    if hasattr(env, 'shutdown'):
        env.shutdown()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
