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

# create_video_wall.py
#
# Combines sim and real recordings into a 11x10 video wall
# Layout: 11 rows, 10 columns (5 games per row: G1_Sim, G1_Real, G2_Sim, G2_Real, ...)

import os
import sys
import cv2
import numpy as np

# Hard-coded video files (relative to recordings directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECORDINGS_DIR = os.path.join(PROJECT_ROOT, 'recordings')
OUTPUT_FILE = os.path.join(RECORDINGS_DIR, 'video_wall.mp4')

# All 53 Atari games (excluding skiing which didn't work)
# Each tuple: (sim_file, real_file, game_name)
GAMES = [
    ('alien_sim.mp4', 'alien_real.mp4', 'Alien'),
    ('amidar_sim.mp4', 'amidar_real.mp4', 'Amidar'),
    ('assault_sim.mp4', 'assault_real.mp4', 'Assault'),
    ('asterix_sim.mp4', 'asterix_real.mp4', 'Asterix'),
    ('asteroids_sim.mp4', 'asteroids_real.mp4', 'Asteroids'),
    ('atlantis_sim.mp4', 'atlantis_real.mp4', 'Atlantis'),
    ('bank_heist_sim.mp4', 'bank_heist_real.mp4', 'Bank Heist'),
    ('battle_zone_sim.mp4', 'battle_zone_real.mp4', 'Battle Zone'),
    ('beam_rider_sim.mp4', 'beam_rider_real.mp4', 'Beam Rider'),
    ('berzerk_sim.mp4', 'berzerk_real.mp4', 'Berzerk'),
    ('bowling_sim.mp4', 'bowling_real.mp4', 'Bowling'),
    ('boxing_sim.mp4', 'boxing_real.mp4', 'Boxing'),
    ('breakout_sim.mp4', 'breakout_real.mp4', 'Breakout'),
    ('centipede_sim.mp4', 'centipede_real.mp4', 'Centipede'),
    ('chopper_command_sim.mp4', 'chopper_command_real.mp4', 'Chopper Command'),
    ('crazy_climber_sim.mp4', 'crazy_climber_real.mp4', 'Crazy Climber'),
    ('defender_sim.mp4', 'defender_real.mp4', 'Defender'),
    ('demon_attack_sim.mp4', 'demon_attack_real.mp4', 'Demon Attack'),
    ('double_dunk_sim.mp4', 'double_dunk_real.mp4', 'Double Dunk'),
    ('enduro_sim.mp4', 'enduro_real.mp4', 'Enduro'),
    ('fishing_derby_sim.mp4', 'fishing_derby_real.mp4', 'Fishing Derby'),
    ('freeway_sim.mp4', 'freeway_real.mp4', 'Freeway'),
    ('frostbite_sim.mp4', 'frostbite_real.mp4', 'Frostbite'),
    ('gopher_sim.mp4', 'gopher_real.mp4', 'Gopher'),
    ('gravitar_sim.mp4', 'gravitar_real.mp4', 'Gravitar'),
    ('hero_sim.mp4', 'hero_real.mp4', 'Hero'),
    ('ice_hockey_sim.mp4', 'ice_hockey_real.mp4', 'Ice Hockey'),
    ('kangaroo_sim.mp4', 'kangaroo_real.mp4', 'Kangaroo'),
    ('krull_sim.mp4', 'krull_real.mp4', 'Krull'),
    ('kung_fu_master_sim.mp4', 'kung_fu_master_real.mp4', 'Kung Fu Master'),
    ('montezuma_revenge_sim.mp4', 'montezuma_revenge_real.mp4', 'Montezuma Revenge'),
    ('ms_pacman_sim.mp4', 'ms_pacman_real.mp4', 'Ms. Pac-Man'),
    ('name_this_game_sim.mp4', 'name_this_game_real.mp4', 'Name This Game'),
    ('phoenix_sim.mp4', 'phoenix_real.mp4', 'Phoenix'),
    ('pitfall_sim.mp4', 'pitfall_real.mp4', 'Pitfall'),
    ('pong_sim.mp4', 'pong_real.mp4', 'Pong'),
    ('private_eye_sim.mp4', 'private_eye_real.mp4', 'Private Eye'),
    ('qbert_sim.mp4', 'qbert_real.mp4', 'Q*bert'),
    ('road_runner_sim.mp4', 'road_runner_real.mp4', 'Road Runner'),
    ('robotank_sim.mp4', 'robotank_real.mp4', 'Robotank'),
    ('seaquest_sim.mp4', 'seaquest_real.mp4', 'Seaquest'),
    # skiing excluded - didn't work
    ('solaris_sim.mp4', 'solaris_real.mp4', 'Solaris'),
    ('space_invaders_sim.mp4', 'space_invaders_real.mp4', 'Space Invaders'),
    ('star_gunner_sim.mp4', 'star_gunner_real.mp4', 'Star Gunner'),
    ('surround_sim.mp4', 'surround_real.mp4', 'Surround'),
    ('tennis_sim.mp4', 'tennis_real.mp4', 'Tennis'),
    ('time_pilot_sim.mp4', 'time_pilot_real.mp4', 'Time Pilot'),
    ('tutankham_sim.mp4', 'tutankham_real.mp4', 'Tutankham'),
    ('venture_sim.mp4', 'venture_real.mp4', 'Venture'),
    ('video_pinball_sim.mp4', 'video_pinball_real.mp4', 'Video Pinball'),
    ('wizard_of_wor_sim.mp4', 'wizard_of_wor_real.mp4', 'Wizard of Wor'),
    ('yars_revenge_sim.mp4', 'yars_revenge_real.mp4', 'Yars Revenge'),
    ('zaxxon_sim.mp4', 'zaxxon_real.mp4', 'Zaxxon')
]

# Layout configuration
GAMES_PER_ROW = 5  # 5 games per row
VIDEOS_PER_ROW = GAMES_PER_ROW * 2  # 10 videos per row (sim + real for each game)
TARGET_WIDTH = 160  # Resize to agent observation size
TARGET_HEIGHT = 210


def create_video_wall():
    """Create a video wall combining all game recordings"""
    print("[create_video_wall] Starting video wall creation...")
    print(f"[create_video_wall] Input directory: {RECORDINGS_DIR}")
    print(f"[create_video_wall] Output file: {OUTPUT_FILE}")
    print(f"[create_video_wall] Layout: {GAMES_PER_ROW} games per row, {len(GAMES)} total games")
    
    # Open all video captures
    video_captures = []
    frame_counts = []
    
    for sim_file, real_file, game_name in GAMES:
        sim_path = os.path.join(RECORDINGS_DIR, sim_file)
        real_path = os.path.join(RECORDINGS_DIR, real_file)
        
        # Check if files exist
        if not os.path.exists(sim_path):
            print(f"[create_video_wall] ERROR: {sim_path} not found!")
            return
        if not os.path.exists(real_path):
            print(f"[create_video_wall] ERROR: {real_path} not found!")
            return
        
        # Open video captures
        sim_cap = cv2.VideoCapture(sim_path)
        real_cap = cv2.VideoCapture(real_path)
        
        if not sim_cap.isOpened():
            print(f"[create_video_wall] ERROR: Could not open {sim_path}")
            return
        if not real_cap.isOpened():
            print(f"[create_video_wall] ERROR: Could not open {real_path}")
            return
        
        video_captures.append((sim_cap, real_cap, game_name))
        
        # Track frame counts
        sim_frames = int(sim_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        real_frames = int(real_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_counts.append(min(sim_frames, real_frames))
        
        print(f"[create_video_wall] Loaded {game_name}: {sim_frames} sim frames, {real_frames} real frames")
    
    # Use minimum frame count across all videos
    total_frames = min(frame_counts)
    print(f"[create_video_wall] Total frames to process: {total_frames}")
    
    # Get dimensions from first video (but we'll resize to TARGET_WIDTH x TARGET_HEIGHT)
    ret, first_frame = video_captures[0][0].read()
    if not ret:
        print("[create_video_wall] ERROR: Could not read first frame")
        return
    video_captures[0][0].set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
    
    original_height, original_width = first_frame.shape[:2]
    print(f"[create_video_wall] Original video size: {original_width}x{original_height}")
    print(f"[create_video_wall] Resizing each video to: {TARGET_WIDTH}x{TARGET_HEIGHT}")
    
    # Calculate wall dimensions (10 columns, 11 rows)
    # 10 videos per row (5 games × 2 videos each: sim, real)
    # 11 rows (53 games ÷ 5 games per row = 10.6, rounded up to 11)
    num_rows = (len(GAMES) + GAMES_PER_ROW - 1) // GAMES_PER_ROW  # Ceiling division
    wall_width = TARGET_WIDTH * VIDEOS_PER_ROW
    wall_height = TARGET_HEIGHT * num_rows
    print(f"[create_video_wall] Video wall size: {wall_width}x{wall_height} ({VIDEOS_PER_ROW} columns × {num_rows} rows)")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    out = cv2.VideoWriter(OUTPUT_FILE, fourcc, fps, (wall_width, wall_height))
    
    if not out.isOpened():
        print("[create_video_wall] ERROR: Could not create output video writer")
        return
    
    print(f"[create_video_wall] Processing frames...")
    
    # Process each frame
    for frame_idx in range(total_frames):
        # Create blank canvas for video wall
        wall_frame = np.zeros((wall_height, wall_width, 3), dtype=np.uint8)
        
        # Read and place each game's frames
        for game_idx, (sim_cap, real_cap, game_name) in enumerate(video_captures):
            # Read frames
            ret_sim, sim_frame = sim_cap.read()
            ret_real, real_frame = real_cap.read()
            
            if not ret_sim or not ret_real:
                print(f"[create_video_wall] Warning: Could not read frame {frame_idx} for {game_name}")
                continue
            
            # Resize frames to target size (210x160) - no labels since videos are small
            sim_frame_resized = cv2.resize(sim_frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                          interpolation=cv2.INTER_LINEAR)
            real_frame_resized = cv2.resize(real_frame, (TARGET_WIDTH, TARGET_HEIGHT), 
                                           interpolation=cv2.INTER_LINEAR)
            
            # Calculate position in wall
            # Layout: 5 games per row, each game has sim (left) and real (right)
            # Row 0: Game0_Sim, Game0_Real, Game1_Sim, Game1_Real, Game2_Sim, Game2_Real, ...
            # Row 1: Game5_Sim, Game5_Real, Game6_Sim, Game6_Real, ...
            # etc.
            row = game_idx // GAMES_PER_ROW
            col_offset = (game_idx % GAMES_PER_ROW) * 2  # Each game takes 2 columns
            
            y_start = row * TARGET_HEIGHT
            y_end = y_start + TARGET_HEIGHT
            
            # Place sim (left within this game's pair)
            x_start_sim = col_offset * TARGET_WIDTH
            x_end_sim = x_start_sim + TARGET_WIDTH
            wall_frame[y_start:y_end, x_start_sim:x_end_sim] = sim_frame_resized
            
            # Place real (right within this game's pair)
            x_start_real = (col_offset + 1) * TARGET_WIDTH
            x_end_real = x_start_real + TARGET_WIDTH
            wall_frame[y_start:y_end, x_start_real:x_end_real] = real_frame_resized
        
        # Write combined frame
        out.write(wall_frame)
        
        # Progress indicator
        if frame_idx % 30 == 0 or frame_idx == total_frames - 1:
            progress = (frame_idx + 1) / total_frames * 100
            print(f"[create_video_wall] Progress: {frame_idx + 1}/{total_frames} ({progress:.1f}%)", end='\r')
    
    print()  # New line after progress
    
    # Release all resources
    for sim_cap, real_cap, _ in video_captures:
        sim_cap.release()
        real_cap.release()
    out.release()
    
    print(f"[create_video_wall] Video wall created successfully!")
    print(f"[create_video_wall] Output saved to: {OUTPUT_FILE}")
    
    # Get file size
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"[create_video_wall] File size: {file_size_mb:.2f} MB")


if __name__ == "__main__":
    create_video_wall()

