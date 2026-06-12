#!/bin/bash
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

# record_all_games.sh
#
# Records 1000 frames for both sim and real environments across all 54 Atari games
# Output: {game}_sim.mp4 and {game}_real.mp4 for each game
#
# WARNING: This will take 3-4 hours to complete!
# Consider running in a screen or tmux session:
#   screen -S recording
#   ./useful_scripts/record_all_games.sh
#   # Press Ctrl-A, then D to detach
#   # Use 'screen -r recording' to reattach

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_ROOT/recordings"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# List of all Atari 57 games to record
GAMES=(
    "alien" "amidar" "assault" "asterix" "asteroids" "atlantis"
    "bank_heist" "battle_zone" "beam_rider" "berzerk" "bowling" "boxing"
    "breakout" "centipede" "chopper_command" "crazy_climber" "defender" "demon_attack"
    "double_dunk" "enduro" "fishing_derby" "freeway" "frostbite" "gopher"
    "gravitar" "hero" "ice_hockey" "kangaroo" "krull" "kung_fu_master"
    "montezuma_revenge" "ms_pacman" "name_this_game" "phoenix" "pitfall" "pong"
    "private_eye" "qbert" "road_runner" "robotank" "seaquest" 
    # "solaris" "space_invaders" "star_gunner" "surround" "tennis" "time_pilot"
    # "tutankham" "venture" "video_pinball" "wizard_of_wor" "yars_revenge" "zaxxon"
)

echo "====================================="
echo "Recording all games (sim and real)"
echo "====================================="
echo "Output directory: $OUTPUT_DIR"
echo ""

# Record each game for both sim and real
for game in "${GAMES[@]}"; do
    echo "-------------------------------------"
    echo "Recording $game"
    echo "-------------------------------------"
    
    # Record sim version
    echo "[1/2] Recording ${game}_sim.mp4..."
    python "$SCRIPT_DIR/visualize_stream.py" \
        --env sim \
        --game "$game" \
        --frames 1000 \
        --output "$OUTPUT_DIR/${game}_sim.mp4" \
        --output-fps 30 \
        --fps 30 \
        --random-policy \
        --no-text
    
    echo "[2/2] Recording ${game}_real.mp4..."
    python "$SCRIPT_DIR/visualize_stream.py" \
        --env real \
        --game "$game" \
        --frames 1000 \
        --output "$OUTPUT_DIR/${game}_real.mp4" \
        --output-fps 30 \
        --random-policy \
        --no-text
    
    echo "✓ Completed $game"
    echo ""
done

echo "====================================="
echo "All recordings complete!"
echo "====================================="
echo "Total games recorded: ${#GAMES[@]}"
echo "Total video files: $((${#GAMES[@]} * 2))"
echo ""
echo "Recorded videos:"
ls -lh "$OUTPUT_DIR"/*.mp4 | head -20
echo "... (showing first 20 files)"
echo ""
echo "Next step: Run create_video_wall.py to combine them"
echo "(Note: The video wall script currently only combines the first 6 games)"

