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
import io
import pickle
import torch

# Add project root to path for clean imports
sys.path.insert(0, os.path.dirname(__file__))

from sim_env import SimEnv, LatencyModel

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
    'fps', 'use_reduced_action_set',
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
    parser.add_argument('--robotroller-config', type=str, default=None,
                        help='Path to robotroller config (overrides checkpoint; used when --env real)')
    parser.add_argument('--comment', type=str, default='no comment', help='Optional comment to log with finetune results')
    args = parser.parse_args()

    model_doc = load_checkpoint(args.checkpoint_dir)

    agent_parms, benchmark_keys = extract_agent_parms(model_doc)
    benchmark_keys.add('use_reduced_action_set')
    
    parms = {k: (typed_value(str(v)) if isinstance(v, str) else v)
             for k, v in agent_parms.items() if k not in benchmark_keys}

    # Create Environment
    print(f"\nCreating environment: {agent_parms['game']}")
    robotroller_config_path = None
    if args.env == 'real':
        from real_env import RealEnv

        print("Using physical Atari environment")
        robotroller_config_path = args.robotroller_config or model_doc.get("robotroller_config_path")
        if not robotroller_config_path:
            print("Error: robotroller config path required when --env real (set --robotroller-config or train with robotroller_config_path in experiment config)")
            sys.exit(1)
        env = RealEnv(
            game=agent_parms['game'],
            seed=agent_parms['seed'],
            latency_model=None,
            max_frames_without_reward=agent_parms['max_frames_without_reward'],
            config_path=robotroller_config_path,
            use_reduced_action_set=agent_parms.get('use_reduced_action_set', False)
        )
        print(f"[finetune_policy] Using robotroller config at {robotroller_config_path}")
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
    if robotroller_config_path is not None:
        finetune_checkpoint['robotroller_config_path'] = robotroller_config_path
        with open(robotroller_config_path, "r") as f:
            finetune_checkpoint['robotroller_config'] = json.load(f)

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
    if robotroller_config_path is not None:
        finetune_results['robotroller_config_path'] = robotroller_config_path
        with open(robotroller_config_path, "r") as f:
            finetune_results['robotroller_config'] = json.load(f)
    save_results_json(os.path.join(args.checkpoint_dir, 'finetune_results.json'), finetune_results)

    if hasattr(env, 'shutdown'):
        env.shutdown()
    print("Done!")

if __name__ == '__main__':
    main()

