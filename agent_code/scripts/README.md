# Scripts

Run these from the `agent_code/` directory.

## visualize_stream.py

Records agent observations from the sim or real environment to an MP4.

```bash
# Sim
python scripts/visualize_stream.py \
  --env sim \
  --game pong \
  --output pong_sim.mp4

# Real (game must be running on the devbox; create physical_atari_sample_config.json first — see agent README Robotroller config)
python scripts/visualize_stream.py \
  --env real \
  --config physical_atari_configs/physical_atari_sample_config.json \
  --game pong \
  --output pong_real.mp4
```

Common flags: `--frames` (default 1000), `--output-fps` (default 30), `--random-policy`, `--no-text`.

## camera_focus_sweep.py

Records a video while sweeping camera focus from 200 to 600 (step 10, 2s per value). Writes `focus_sweep.mp4`.

```bash
python scripts/camera_focus_sweep.py \
  --config-path physical_atari_configs/robotroller.default.json
```
