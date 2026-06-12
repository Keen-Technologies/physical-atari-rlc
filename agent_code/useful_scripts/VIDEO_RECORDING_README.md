# Video Recording Scripts

Scripts to record and combine videos from both simulated and real Atari environments.

## Files

- `record_all_games.sh` - Records 1000 frames for 5 games in both sim and real environments
- `create_video_wall.py` - Combines all recordings into a 5x2 video wall
- `visualize_stream.py` - Core recording script (used by record_all_games.sh)

## Games Recorded

All 54 Atari games including:
- alien, amidar, assault, asterix, asteroids, atlantis
- bank_heist, battle_zone, beam_rider, berzerk, bowling, boxing
- breakout, centipede, chopper_command, crazy_climber, defender, demon_attack
- double_dunk, enduro, fishing_derby, freeway, frostbite, gopher
- gravitar, hero, ice_hockey, kangaroo, krull, kung_fu_master
- montezuma_revenge, ms_pacman, name_this_game, phoenix, pitfall, pong
- private_eye, qbert, road_runner, robotank, seaquest, skiing
- solaris, space_invaders, star_gunner, surround, tennis, time_pilot
- tutankham, venture, video_pinball, wizard_of_wor, yars_revenge, zaxxon

## Usage

### Step 1: Record All Games

Run the bash script to record all 10 videos (5 games × 2 environments):

```bash
cd /path/to/robotroller_experiments
./useful_scripts/record_all_games.sh
```

This will:
- Create a `recordings/` directory in the project root
- Record 1000 frames for each game in both sim and real
- Generate **108 MP4 files** (54 games × 2 environments)
  - `{game}_sim.mp4` and `{game}_real.mp4` for each game

**Note**: Recording all 54 games will take a **VERY long time**:
- Sim environment: ~30-40 seconds per game → ~30-40 minutes total
- Real environment: ~2-3 minutes per game (includes game switching) → ~2-3 hours total
- **Total estimated time: 3-4 hours** for all recordings

### Step 2: Create Video Wall

After all recordings are complete, combine them into a video wall:

```bash
python useful_scripts/create_video_wall.py
```

This will create `recordings/video_wall.mp4` with:
- Layout: 11 rows × 10 columns
- Each row shows 5 games (10 videos: G1_Sim, G1_Real, G2_Sim, G2_Real, ...)
- Each video resized to 160×210 (agent observation size)
- Total resolution: 1600×2310 pixels

## Output

### Individual Videos
- Located in: `recordings/`
- Format: MP4 (H.264)
- FPS: 30
- Frames: 1000 each
- Size: ~640×840 per video (upscaled 4× from 160×210 agent observation)
- Note: Videos are recorded at 640×840 but resized to 160×210 when placed in the wall

### Video Wall
- Located in: `recordings/video_wall.mp4`
- Format: MP4 (H.264)
- FPS: 30
- Frames: 1000
- Size: 1600×2310 (10 columns × 11 rows)
- Individual video size: 160×210 (agent observation size)
- Layout: **53 games** arranged as 5 games per row (10 videos per row: sim + real)
  - All games from Alien to Zaxxon (excluding Skiing)
  - Each game shows sim (left) and real (right) side-by-side
  ```
  Row  1: Alien_S, Alien_R, Amidar_S, Amidar_R, Assault_S, Assault_R, ...
  Row  2: Atlantis_S, Atlantis_R, Bank Heist_S, Bank Heist_R, ...
  ...
  Row 11: Wizard_S, Wizard_R, Yars_S, Yars_R, Zaxxon_S, Zaxxon_R
  ```

### File Layout
```
recordings/
├── alien_sim.mp4
├── alien_real.mp4
├── amidar_sim.mp4
├── amidar_real.mp4
├── ... (106 total video files for 53 games, excluding skiing)
├── zaxxon_sim.mp4
├── zaxxon_real.mp4
└── video_wall.mp4  (created by create_video_wall.py)
```

**Note**: The video wall script combines all 53 games (excluding Skiing which didn't work). Each video is resized to 160×210 to fit in the wall. The final wall is 1600×2310 pixels.

## Customization

### Record Only a Subset of Games

To record only specific games (recommended for testing), edit `record_all_games.sh`:

```bash
# Example: Record only 5 games for quick testing
GAMES=("pong" "breakout" "space_invaders" "ms_pacman" "seaquest")
```

Or use the full list of all 54 games (default):

```bash
GAMES=(
    "alien" "amidar" "assault" "asterix" "asteroids" "atlantis"
    # ... (see script for full list)
)
```

**create_video_wall.py:**
```python
# All 53 games (excluding skiing)
GAMES = [
    ('alien_sim.mp4', 'alien_real.mp4', 'Alien'),
    ('amidar_sim.mp4', 'amidar_real.mp4', 'Amidar'),
    # ... (see script for full list of 53 games)
    ('zaxxon_sim.mp4', 'zaxxon_real.mp4', 'Zaxxon')
]

# Layout: 5 games per row = 10 videos per row
# Videos resized to 160×210 each
```

### Modify Recording Parameters

Edit `record_all_games.sh` to change:
- `--frames 1000` - number of frames to record
- `--output-fps 30` - output video FPS
- `--fps 30` - simulator FPS (real environment ignores this, set to 30 to match real camera FPS)
- `--random-policy` - use random policy (samples action from 0-17 every 4 frames)
- `--no-text` - disable text overlays (reward, frame count, etc.) on video frames

### Policy Options

By default, the recording script uses a **random policy**:
- Samples a random action from 0-17 (all 18 Atari actions)
- Holds each action for 4 frames before sampling a new one
- This creates more interesting/varied gameplay footage

To record with a **NOOP (do nothing) policy** instead:
- Remove the `--random-policy` flag from `record_all_games.sh`

### Text Overlay Options

By default, the recording script uses `--no-text` to disable text overlays:
- No reward display
- No frame counter
- No timestamps
- Clean frames suitable for video walls

To record **with text overlays** (for debugging or single video viewing):
- Remove the `--no-text` flag from `record_all_games.sh`
- Text will show: current reward, cumulative reward, frame number, timestamp, and episode status

## Requirements

- Python 3 with:
  - OpenCV (`cv2`)
  - NumPy
  - All dependencies from `requirements.txt`
- For real environment recordings:
  - Physical Atari hardware setup
  - Game launched manually on devbox (`PhysicalALE ./games/ <game>`)

## Troubleshooting

### "Video file not found" error
- Make sure `record_all_games.sh` completed successfully
- Check that all 10 MP4 files exist in `recordings/`

### Recording takes very long
- Real environment recordings include game switching delays (10-15 seconds per game)
- **Total time estimate: 3-4 hours for all 54 games**
- Consider running overnight or in a screen/tmux session
- You can modify the GAMES array in `record_all_games.sh` to record only a subset of games

### Video wall quality issues
- Individual videos are upscaled 4× from agent observations (160×210)
- This is intentional to show what the agent actually sees
- Original resolution may appear pixelated when viewed at large sizes

