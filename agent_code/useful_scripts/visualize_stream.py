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

# visualize_stream.py
#
# Records agent observations from RealEnv or SimEnv with rewards displayed.
# This captures what the agent actually sees (processed 160x210 RGB frames).
#

import os
import sys
import argparse
import time
import json
import numpy as np
import cv2

# Add project root to path for clean imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sim_env import SimEnv


def main():
    parser = argparse.ArgumentParser(description="Record agent observations from environment")
    parser.add_argument("--env", type=str, default="real", choices=["real", "sim"],
                        help="Environment type: 'real' for physical hardware, 'sim' for simulator (default: real)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to robotroller config file (required for real environment)")
    parser.add_argument("--game", type=str, default="pong",
                        help="Game name (e.g., pong, breakout)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--frames", type=int, default=1000, 
                        help="Number of frames to record")
    parser.add_argument("--output", type=str, default="agent_recording.mp4", 
                        help="Output video filename")
    parser.add_argument("--output-fps", type=int, default=30,
                        help="Output video FPS")
    parser.add_argument("--max-frames-without-reward", type=int, default=18000,
                        help="Max frames without reward before truncation (default: 18000)")
    parser.add_argument("--fps", type=int, default=60,
                        help="FPS for sim environment (default: 60, must divide 60 evenly, ignored for real)")
    parser.add_argument("--random-policy", action="store_true",
                        help="Use random policy (sample action from 0-17 every 4 frames)")
    parser.add_argument("--no-text", action="store_true",
                        help="Disable text overlays on frames (no reward, frame count, etc.)")
    
    args = parser.parse_args()

    # Initialize environment
    print(f"[visualize_stream] Initializing {args.env} environment for game: {args.game}")
    
    if args.env == "real":
        from real_env import RealEnv

        print(f"[visualize_stream] Loading configuration from: {args.config}")
        config_path = os.path.expanduser(args.config)
        env = RealEnv(
            game=args.game,
            seed=args.seed,
            latency_model=None,
            max_frames_without_reward=args.max_frames_without_reward,
            config_path=config_path
        )
    else:
        # Use simulated environment
        print(f"[visualize_stream] Using simulated environment (no config needed)")
        print(f"[visualize_stream] FPS: {args.fps}")
        env = SimEnv(
            game=args.game,
            seed=args.seed,
            latency_model=None,
            max_frames_without_reward=args.max_frames_without_reward,
            fps=args.fps
        )
    
    print(f"[visualize_stream] Environment initialized, waiting for first frame...")

    # Reset environment to get initial observation
    env.reset()
    
    # Wait for environment to stabilize
    time.sleep(1.0)

    # Get first frame to verify environment is working
    first_obs = env.get_observation()
    if first_obs is None or first_obs.size == 0:
        print(f"[visualize_stream] ERROR: Could not get observation from environment")
        return

    height, width, channels = first_obs.shape  # Should be 210x160x3
    print(f"[visualize_stream] Agent observation size: {width}x{height}x{channels}")

    # Initialize Video Writer (upscale for better visibility)
    scale_factor = 4  # Make video 640x840 for better viewing
    output_width = width * scale_factor
    output_height = height * scale_factor
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.output_fps, (output_width, output_height))

    print(f"[visualize_stream] Recording {args.frames} frames at {args.output_fps} FPS")
    print(f"[visualize_stream] Output video size: {output_width}x{output_height}")

    # Record frames
    frames_data = []  # Store (frame, reward, terminated, truncated) tuples
    start_time = time.time()
    total_reward = 0
    action = 0  # NOOP action
    
    # Random policy setup
    if args.random_policy:
        print(f"[visualize_stream] Using random policy (sampling every 4 frames from 0-17)")
        action_hold_frames = 4
        frames_since_last_sample = 0

    try:
        for frame_idx in range(args.frames):
            # Sample new action if using random policy and it's time
            if args.random_policy:
                if frames_since_last_sample % action_hold_frames == 0:
                    action = np.random.randint(0, 18)  # Sample from 0-17
                frames_since_last_sample += 1
            
            # Step environment with current action
            env.step(action)
            
            # Get observation and reward
            obs = env.get_observation()
            reward = env.get_reward()
            if reward != 0:
                print("Reward is not zero", reward)
            terminated = env.get_terminated()
            truncated = env.get_truncated()
            
            total_reward += reward
            
            # Store frame data
            if obs is not None and obs.size > 0:
                frames_data.append((obs.copy(), reward, terminated, truncated))

            # Print progress
            if frame_idx % 30 == 0:
                elapsed = time.time() - start_time
                capture_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
                print(f"[visualize_stream] Captured {frame_idx + 1}/{args.frames} frames "
                      f"({elapsed:.1f}s, {capture_fps:.1f} FPS, reward: {total_reward})", end='\r')
            
            # Reset if episode ended
            if terminated or truncated:
                print(f"\n[visualize_stream] Episode ended at frame {frame_idx + 1}, resetting...")
                env.reset()
                total_reward = 0
                if args.random_policy:
                    frames_since_last_sample = 0
                    action = 0  # Reset to NOOP

    except KeyboardInterrupt:
        print("\n[visualize_stream] Capture interrupted by user")
        # Cleanup environment before exiting
        if hasattr(env, 'shutdown'):
            env.shutdown()
        return
    
    capture_time = time.time() - start_time
    print(f"\n[visualize_stream] Captured {len(frames_data)} frames in {capture_time:.2f}s")

    if len(frames_data) == 0:
        print("[visualize_stream] No frames captured, exiting")
        return

    # Process and write frames to video
    print(f"[visualize_stream] Writing frames to {args.output}...")
    process_start_time = time.time()
    cumulative_reward = 0

    for frame_idx, (obs, reward, terminated, truncated) in enumerate(frames_data):
        cumulative_reward += reward
        
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        
        # Upscale frame for better visibility
        frame_large = cv2.resize(frame_bgr, (output_width, output_height), 
                                interpolation=cv2.INTER_NEAREST)
        
        # Add text overlays (unless --no-text flag is set)
        if not args.no_text:
            timestamp = frame_idx / args.output_fps
            
            # Current reward (highlight if non-zero)
            reward_color = (0, 255, 0) if reward > 0 else (0, 0, 255) if reward < 0 else (200, 200, 200)
            reward_text = f"Reward: {reward:+d}"
            cv2.putText(frame_large, reward_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, reward_color, 2, cv2.LINE_AA)
            
            # Cumulative reward
            cv2.putText(frame_large, f"Total: {cumulative_reward:+d}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Frame info
            cv2.putText(frame_large, f"Frame: {frame_idx}/{len(frames_data)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Time
            cv2.putText(frame_large, f"Time: {timestamp:.2f}s", (10, output_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Episode status
            if terminated or truncated:
                status_text = "TERMINATED" if terminated else "TRUNCATED"
                cv2.putText(frame_large, status_text, (output_width // 2 - 100, output_height // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)

        out.write(frame_large)

        if frame_idx % 30 == 0:
            elapsed = time.time() - process_start_time
            process_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            print(f"[visualize_stream] Processed {frame_idx + 1}/{len(frames_data)} frames "
                  f"({elapsed:.1f}s, {process_fps:.1f} FPS)", end='\r')

    out.release()
    process_time = time.time() - process_start_time
    total_time = capture_time + process_time
    
    print(f"\n[visualize_stream] Processing complete!")
    print(f"[visualize_stream] Capture: {capture_time:.2f}s ({len(frames_data)/capture_time:.1f} FPS)")
    print(f"[visualize_stream] Processing: {process_time:.2f}s ({len(frames_data)/process_time:.1f} FPS)")
    print(f"[visualize_stream] Total: {total_time:.2f}s")
    print(f"[visualize_stream] Final cumulative reward: {cumulative_reward}")
    print(f"[visualize_stream] Saved video to {args.output}")
    
    # Cleanup environment if it has a shutdown method
    if hasattr(env, 'shutdown'):
        env.shutdown()


if __name__ == "__main__":
    main()
