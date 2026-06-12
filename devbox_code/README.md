# Devbox Setup

Runs on a Raspberry Pi devbox: builds and runs the Stella-based Atari emulator (`PhysicalALE`) with AprilTag overlays and GPIO input.

## Prerequisites

- Raspberry Pi OS Full (Debian 13 "trixie") — fresh install
- Clone this repository, then work from the `devbox_code` directory

## Install dependencies

```sh
sudo apt update
sudo apt install -y libsdl2-dev libgpiod-dev cmake build-essential pkg-config
```

## Build and install

```sh
cd devbox_code
mkdir -p build && cd build
cmake ..
make -j"$(nproc)"
sudo make install   # installs PhysicalALE to /usr/local/bin
```

## ROM files

We do not ship ROM files in this repository. You must place supported Atari 2600 ROMs in a directory (e.g. `./games/`) before running `PhysicalALE`.

The easiest way to get them is via [ale_py](https://github.com/Farama-Foundation/Arcade-Learning-Environment), which bundles supported ROMs in the pip package (ale-py 0.11+):

```sh
pip install ale-py
mkdir -p games
python3 -c "from ale_py import roms; import shutil; from pathlib import Path; d=Path('games'); [shutil.copy(p, d/f'{n}.bin') for n in roms.get_all_rom_ids() if (p:=roms.get_rom_path(n))]"
```

On Raspberry Pi OS, if `pip install` fails because system Python is externally managed, use a virtual environment or pass `--break-system-packages` to `pip install ale-py`.

## Run

```sh
cd devbox_code
PhysicalALE ./games/ pong
PhysicalALE ./games/ pong my_experiment   # optional override for results filename
```

Usage: `PhysicalALE <RomDirectory> <game_name> [results_filename]`

On shutdown, results are saved locally to `{game_name}_{timestamp}.json` (UTC). Pass an optional third argument to override the filename (saved as `{results_filename}.json`).

## Keyboard controls

Requires a USB keyboard connected to the Pi.

- `T`: Toggle corner AprilTag visibility
- `F`: Toggle FPS counter
- `Q`: Quit
