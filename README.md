# Physical Atari RLC

Software for training RL agents on a physical Atari setup: a **Devbox** (Raspberry Pi emulator with AprilTag overlays) and a **Robotroller** (camera + servo-driven joystick) controlled from an agent machine.

## Overview

- **Devbox** (Raspberry Pi): runs `PhysicalALE`, displays the game and AprilTags on a Waveshare screen.
- **Agent machine** (Linux): camera observes the Devbox screen, Robotroller servos press the joystick, and `agent_code` runs training/evaluation.

## Setup

### 1. Build Robotroller

- Follow the [Robotroller assembly guide](https://keenagi.com/research/physical-atari/robotroller.html).
- Configure servo IDs and baud rate before first use (see [Hardware setup](#hardware-setup) below).

### 2. Set up the Devbox

The microSD card cannot be inserted or removed once the chassis is built. Order:

1. Flash **Raspberry Pi OS Full** (64-bit, Debian 13 "trixie") and insert the SD card — see [devbox_code/README.md](devbox_code/README.md#flash-raspberry-pi-os).
2. Assemble the Devbox — follow the [Devbox assembly guide](https://keenagi.com/research/physical-atari/devbox.html).
3. Clone this repository, install dependencies, and build `PhysicalALE` (see [devbox_code/README.md](devbox_code/README.md)).
4. Place Atari 2600 ROMs in a directory (e.g. `devbox_code/games/`).
5. Launch a game: `PhysicalALE ./games/ pong`

### 3. Set up the agent machine

On the Linux machine with the camera and Robotroller:

1. Build and install the I/O library (see [input_output_cpp_library/README.md](input_output_cpp_library/README.md)).
2. Install Python dependencies:
   ```bash
   cd agent_code
   pip install -r requirements.txt
   ```
3. Create a local Robotroller config (see [agent_code/README.md#robotroller-config](agent_code/README.md#robotroller-config)):
   ```bash
   cp physical_atari_configs/robotroller.default.json physical_atari_configs/physical_atari_sample_config.json
   ```
   Edit `physical_atari_sample_config.json` for your camera device and serial port.

### 4. Run a random agent

With the Devbox running a game:

```bash
# On Devbox
PhysicalALE ./games/ pong

# On agent machine (from agent_code/)
python learn_policy.py --config experiment_configs/agent_random_real.json --run 0
```

For sim-only testing (no hardware), see [agent_code/README.md](agent_code/README.md).

## Hardware setup

Before using the physical environment, configure each Dynamixel XC330 servo. Factory defaults are ID **1** and baud rate **57600**. The Robotroller expects three servos on the same bus at baud rate **1,000,000**:

| ID | Role |
|----|------|
| 50 | Fire button |
| 51 | Left/right D-pad |
| 52 | Up/down D-pad |

Configure one servo at a time (only one device on the bus during programming):

1. Connect a USB serial adapter to the **agent machine** (not the Devbox).
2. Install dependencies: `pip install -r setup_scripts/requirements.txt`
3. Find the serial port: `ls /dev/serial/by-id/`
4. Connect **one** servo and power it.
5. Assign the target ID with [`setup_scripts/change_id.py`](setup_scripts/change_id.py) (use `--baud_rate 57600` for factory-default servos).
6. Set baud rate to 1,000,000 with [`setup_scripts/change_baud_rate.py`](setup_scripts/change_baud_rate.py).
7. Disconnect that servo, connect the next, and repeat.

Set `SERIAL` to your device path. Example for the fire-button servo:

```bash
# Assign ID 50 (communicate at factory baud 57600)
python setup_scripts/change_id.py --path "$SERIAL" --current_id 1 --new_id 50 --baud_rate 57600

# Switch servo to 1 Mbps (required by the Robotroller library)
python setup_scripts/change_baud_rate.py --path "$SERIAL" --id 50 --current_baud_rate 57600 --new_baud_rate 1000000
```

Repeat for the left/right servo (ID 51) and up/down servo (ID 52).

Once all three servos are configured, connect them together on the same bus. Set `robot.serial_port` in your Robotroller config — see [agent_code/README.md#robotroller-config](agent_code/README.md#robotroller-config).

## Repository layout

| Directory | Purpose |
|-----------|---------|
| [`agent_code/`](agent_code/) | RL training, evaluation, and finetuning ([README](agent_code/README.md)) |
| [`devbox_code/`](devbox_code/) | Devbox emulator (`PhysicalALE`) ([README](devbox_code/README.md)) |
| [`input_output_cpp_library/`](input_output_cpp_library/) | Camera, AprilTag, and servo I/O (`robotroller` module) ([README](input_output_cpp_library/README.md)) |
| [`setup_scripts/`](setup_scripts/) | Dynamixel servo ID and baud rate configuration |
