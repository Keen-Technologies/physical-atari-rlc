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

# finetune_policy.py
#
# Script to load a trained model, evaluate it, fine-tune it, and evaluate again.
#
# Usage: python finetune_policy.py --checkpoint-dir ./checkpoints/<run_name> --learning_steps 100000
#
# Three phases:
# 1. Pre-evaluation: train=True, update_weights=False (controlled by --eval_before_steps)
# 2. Learning: train=True, update_weights=True (controlled by --learning_steps)
# 3. Post-evaluation: train=False (controlled by --eval_after_steps)

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
    make_run_dir,
    save_checkpoint,
    save_results_json,
)

def typed_value(s):
    """Convert string to appropriate type"""
    try:
        int_val = int(s)
        if '.' in s or 'e' in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s

def run_phase(env, agent, steps, train, update_weights, description):
    """Run a phase of evaluation or training"""
    print(f"\n" + "=" * 50)
    print(f"Starting {description} ({steps} steps, train={train}, update_weights={update_weights})")
    print("=" * 50)

    episode_scores = []
    running_episode_score = 0
    cumulative_reward = 0
    
    # For training stats
    avg_reward_history = []
    moving_avg_reward_history = []
    moving_avg_reward = 0
    log_interval = 100  
    
    episode_start_frame = 0
    episode_start_frame_time = time.time()
    start_time = time.time()
    last_print_time = start_time
    print_interval = 2.0

    # Reset env but keep agent state
    env.reset()
    agent_action_index = 0

    for frame in range(steps):
        # Perceive environment first (matches evaluate_policy structure)
        env.perceive()

        # Get results
        observation = env.get_observation()
        reward = env.get_reward()
        terminated = env.get_terminated()
        truncated = env.get_truncated()

        # Update stats
        running_episode_score += reward
        cumulative_reward += reward
        if train and update_weights:
            moving_avg_reward = 0.999 * moving_avg_reward + 0.001 * reward

        # Logging
        if frame > 0 and frame % log_interval == 0 and train and update_weights:
            avg_reward = cumulative_reward / frame
            avg_reward_history.append((avg_reward, frame))
            moving_avg_reward_history.append((moving_avg_reward, frame))

        # Periodic printing
        current_time = time.time()
        if current_time - last_print_time >= print_interval:
            elapsed = current_time - start_time
            fps = (frame + 1) / elapsed if elapsed > 0 else 0
            if frame % 1000 == 0: # Avoid spamming too much
                 print(f"[{description}] Frame {frame}/{steps}: FPS: {fps:.2f}")
            last_print_time = current_time

        # End of episode handling
        end_of_episode = 0
        if terminated:
            end_of_episode = 2
        elif truncated:
            end_of_episode = 3
        
        if end_of_episode > 1:
            if frame > 0:
                episode_scores.append(running_episode_score)
                running_episode_score = 0
                
                # Stats
                now = time.time()
                frames = frame - episode_start_frame
                fps = frames / (now - episode_start_frame_time)
                episode_start_frame_time = now
                
                # Recent average
                recent_avg = np.mean(episode_scores[-10:]) if episode_scores else 0
                
                print(f"{description} frame:{frame:7} {fps:4.0f}/s ep: {len(episode_scores)-1:3}, {frames:5}={int(episode_scores[-1]):5} ep_avg:{recent_avg:7.1f}")
            
            env.reset()
            episode_start_frame = frame

        # Agent step (matches evaluate_policy structure)
        agent_action_index = agent.get_action(observation, reward, end_of_episode, train=train)
        env.step(agent_action_index)
        agent.learn(observation, reward, end_of_episode, train=train, update_weights=update_weights)

    avg_reward_per_step = cumulative_reward / steps if steps > 0 else 0
    
    results = {
        'episode_scores': episode_scores,
        'avg_reward_per_step': avg_reward_per_step,
        'total_episodes': len(episode_scores),
        'mean_episode_score': np.mean(episode_scores) if episode_scores else 0
    }
    
    if train and update_weights:
        results['avg_reward_history'] = avg_reward_history
        results['moving_avg_reward_history'] = moving_avg_reward_history
        
    return results

def main():
    parser = argparse.ArgumentParser(description='Finetune a trained model')
    parser.add_argument('--checkpoint-dir', type=str, required=True, help='Path to source checkpoint run directory')
    parser.add_argument('--eval_before_steps', type=int, default=18000, help='Number of evaluation steps before training (default: 18000, ~5 min @ 60fps)')
    parser.add_argument('--learning_steps', type=int, required=True, help='Number of learning/training steps')
    parser.add_argument('--eval_after_steps', type=int, default=18000, help='Number of evaluation steps after training (default: 18000, ~5 min @ 60fps)')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu, cuda, mps)')
    parser.add_argument('--env', type=str, default='sim', help='Environment type (sim or real)')
    parser.add_argument('--comment', type=str, default='no comment', help='Optional comment to log with finetune results')
    args = parser.parse_args()

    model_doc = load_checkpoint(args.checkpoint_dir)

    agent_parms, benchmark_keys = extract_agent_parms(model_doc)
    benchmark_keys.add('use_reduced_action_set')
    
    parms = {k: (typed_value(str(v)) if isinstance(v, str) else v)
             for k, v in agent_parms.items() if k not in benchmark_keys}

    # Create Environment
    print(f"\nCreating environment: {agent_parms['game']}")
    if args.env == 'real':
        from real_env import RealEnv

        print("Using physical Atari environment")
        env = RealEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=None,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            config_path=os.path.expanduser("~/robotroller.conf"),
            use_reduced_action_set=agent_parms.get('use_reduced_action_set', False)
        )
    elif args.env == 'sim':
        print("Using simulated Atari environment")
        env = SimEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=LatencyModel(agent_parms['physical']) if agent_parms.get('physical') else None,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            use_reduced_action_set=agent_parms.get('use_reduced_action_set', False)
        )
    else:
        print(f"Error: unknown env type '{args.env}' (expected 'sim' or 'real')")
        sys.exit(1)

    num_actions = env.get_num_actions()

    # Create Agent
    print(f"Creating agent: {agent_parms['module_name']}")
    # Note: For fine-tuning, we might want to use the original training steps or the learning steps 
    # as the 'steps' argument to the agent. Usually this affects scheduling (epsilon decay etc).
    # Since we are fine-tuning, we probably want the agent to think it's continuing or starting a new phase.
    # Passing learning_steps might reset schedules. 
    # However, we load the state later.
    total_steps = args.eval_before_steps + args.learning_steps + args.eval_after_steps
    agent = importlib.import_module(agent_parms['module_name']).Agent(
        agent_parms['seed'],
        num_actions,
        total_steps, 
        args.device,
        **parms
    )

    apply_checkpoint_to_agent(agent, args.checkpoint_dir, args.device, mode='finetune')

    # 1. Pre-Finetune Evaluation (train=True, update_weights=False)
    pre_results = run_phase(env, agent, args.eval_before_steps, train=True, update_weights=False, description="Pre-Finetune Eval")

    # 2. Fine-tuning (train=True, update_weights=True)
    train_results = run_phase(env, agent, args.learning_steps, train=True, update_weights=True, description="Fine-tuning")

    # 3. Post-Finetune Evaluation (train=False)
    post_results = run_phase(env, agent, args.eval_after_steps, train=False, update_weights=False, description="Post-Finetune Eval")

    print("\n" + "=" * 50)
    print("Saving fine-tuned checkpoint...")
    print("=" * 50)

    finetune_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    source_run_name = model_doc.get('run_name', 'Unknown')
    finetune_run_name = f"{source_run_name}__finetune__{finetune_timestamp}"

    checkpoint_dir = model_doc.get('checkpoint_dir', os.path.dirname(args.checkpoint_dir))

    finetune_run_dir = make_run_dir(checkpoint_dir, finetune_run_name)

    finetune_checkpoint = {
        **agent_parms,
        'run_name': finetune_run_name,
        'source_checkpoint_dir': args.checkpoint_dir,
        'source_run_name': source_run_name,
        'finetune_timestamp': finetune_timestamp,
        'eval_before_steps': args.eval_before_steps,
        'learning_steps': args.learning_steps,
        'eval_after_steps': args.eval_after_steps,
        'pre_finetune_episode_scores': pre_results['episode_scores'],
        'pre_finetune_mean_score': pre_results['mean_episode_score'],
        'pre_finetune_avg_reward_per_step': pre_results['avg_reward_per_step'],
        'training_episode_scores': train_results['episode_scores'],
        'training_avg_reward_history': train_results.get('avg_reward_history', []),
        'training_moving_avg_reward_history': train_results.get('moving_avg_reward_history', []),
        'training_lifetime_avg_reward_per_step': train_results['avg_reward_per_step'],
        'post_finetune_episode_scores': post_results['episode_scores'],
        'post_finetune_mean_score': post_results['mean_episode_score'],
        'post_finetune_avg_reward_per_step': post_results['avg_reward_per_step'],
        'comment': args.comment,
        'store_weights': True,
    }

    save_checkpoint(finetune_run_dir, finetune_checkpoint, agent, store_weights=True)

    finetune_results = {
        'finetune_run_name': finetune_run_name,
        'finetune_run_dir': finetune_run_dir,
        'source_checkpoint_dir': args.checkpoint_dir,
        'source_run_name': source_run_name,
        'finetune_timestamp': finetune_timestamp,
        'eval_before_steps': args.eval_before_steps,
        'learning_steps': args.learning_steps,
        'eval_after_steps': args.eval_after_steps,
        'hyperparameters': agent_parms,
        'pre_finetune_episode_scores': pre_results['episode_scores'],
        'pre_finetune_mean_score': pre_results['mean_episode_score'],
        'pre_finetune_avg_reward_per_step': pre_results['avg_reward_per_step'],
        'training_episode_scores': train_results['episode_scores'],
        'training_avg_reward_history': train_results.get('avg_reward_history', []),
        'training_moving_avg_reward_history': train_results.get('moving_avg_reward_history', []),
        'training_lifetime_avg_reward_per_step': train_results['avg_reward_per_step'],
        'post_finetune_episode_scores': post_results['episode_scores'],
        'post_finetune_mean_score': post_results['mean_episode_score'],
        'post_finetune_avg_reward_per_step': post_results['avg_reward_per_step'],
        'comment': args.comment,
    }
    save_results_json(os.path.join(args.checkpoint_dir, 'finetune_results.json'), finetune_results)

    if hasattr(env, 'shutdown'):
        env.shutdown()
    print("Done!")

if __name__ == '__main__':
    main()

