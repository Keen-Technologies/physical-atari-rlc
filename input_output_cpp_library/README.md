# input_output_cpp_library

I/O layer for the physical Atari setup: camera capture, AprilTag detection, and Robotroller servo actions. The primary deliverable is the `robotroller` Python module, which exposes `PhysicalAtariEnv` (used by RL training), `Camera`, and `Robotroller`.

**Prerequisites:** Build the Robotroller hardware, assign servo IDs 50, 51, and 52, and set baud rate to 1,000,000 — see [Hardware setup](../README.md#hardware-setup) in the root README.

For hardware config, tests, and training scripts, see [agent_code/README.md](../agent_code/README.md).

## 1. Install dependencies

### Ubuntu

```bash
sudo apt update
sudo apt install -y build-essential cmake git libopencv-dev python3-dev python3-pybind11 libapriltag-dev
```

### Dynamixel SDK

```bash
git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
cd DynamixelSDK/c++
mkdir build && cd build
cmake ..
make
sudo make install
cd ../../..
```

### Miniconda (optional)

For Python environment management:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

## 2. Build and install

```bash
cd input_output_cpp_library
mkdir -p build && cd build
cmake ..
make
sudo make install
```

Verify the module is available:

```bash
python3 -c "import robotroller; print(robotroller.__doc__)"
```

## License

Copyright 2026 Keen Technologies, Inc.

Licensed under the Apache License, Version 2.0. See the [repository LICENSE](../LICENSE) for the full text.
