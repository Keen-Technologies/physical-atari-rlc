#!/usr/bin/env python3
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

"""
camera_focus_sweep.py - Record video while sweeping through focus values

Records a ~80 second video while changing focus from 200 to 600 in steps of 10.
Focus changes every 2 seconds, allowing visual inspection of different focus values.
"""

import os
import sys
import time
import json
import argparse
import cv2

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'input_output_cpp_library'))

import robotroller


def main():
    parser = argparse.ArgumentParser(description="Record video while sweeping through focus values")
    parser.add_argument(
        "--config-path",
        type=str,
        required=True,
        help="Path to robotroller config file",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Focus Calibration - Video Recording")
    print("=" * 60)
    
    # Load configuration
    config_path = os.path.expanduser(args.config_path)
    print(f"\nLoading config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Extract camera parameters
    camera_device_str = config["camera"]["device"]
    if isinstance(camera_device_str, str) and "/dev/video" in camera_device_str:
        camera_index = int(camera_device_str.split("/dev/video")[-1])
    elif isinstance(camera_device_str, int):
        camera_index = camera_device_str
    else:
        camera_index = int(camera_device_str)
    
    camera_width = config["camera"]["width"]
    camera_height = config["camera"]["height"]
    camera_zoom = config["camera"].get("zoom", 100)
    camera_fps = config["camera"].get("fps", 30)
    camera_exposure = config["camera"].get("exposure", 20)
    camera_brightness = config["camera"].get("brightness", 128)
    camera_contrast = config["camera"].get("contrast", 128)
    target_fps = config["camera"]["target_fps"]
    
    # Focus sweep parameters
    focus_start = 200
    focus_end = 600
    focus_step = 10
    change_interval = 2.0  # Change focus every 2 seconds
    
    focus_values = list(range(focus_start, focus_end + 1, focus_step))
    total_duration = len(focus_values) * change_interval
    
    print(f"\nCamera: {camera_width}x{camera_height} @ {target_fps}fps")
    print(f"Focus sweep: {focus_start} to {focus_end} in steps of {focus_step}")
    print(f"Total focus values: {len(focus_values)}")
    print(f"Change interval: {change_interval}s")
    print(f"Total duration: {total_duration:.0f}s")
    
    # Initialize camera
    print(f"\nInitializing camera...")
    camera = robotroller.Camera(
        camera_index,
        camera_width,
        camera_height,
        focus_start,  # Start with first focus value
        camera_zoom,
        camera_fps,
        camera_exposure,
        camera_brightness,
        camera_contrast
    )
    camera.start()
    
    # Wait for camera to warm up
    print("Camera warming up...")
    time.sleep(1.0)
    
    # Get first frame to determine video properties
    first_frame = camera.getFrame()
    if first_frame is None or first_frame.size == 0:
        print("ERROR: Could not get frame from camera")
        camera.stop()
        return
    
    height, width, channels = first_frame.shape
    
    # Initialize video writer
    output_file = "focus_sweep.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, target_fps, (width, height))
    
    print(f"\nRecording to: {output_file}")
    print("=" * 60)
    
    start_time = time.time()
    last_focus_change_time = start_time
    focus_index = 0
    current_focus = focus_values[focus_index]
    frames_written = 0
    
    try:
        while time.time() - start_time < total_duration:
            # Check if it's time to change focus
            current_time = time.time()
            elapsed = current_time - start_time
            
            if current_time - last_focus_change_time >= change_interval:
                focus_index += 1
                if focus_index < len(focus_values):
                    current_focus = focus_values[focus_index]
                    camera.setFocus(current_focus)
                    last_focus_change_time = current_time
                    print(f"[{elapsed:.1f}s] Focus changed to: {current_focus}")
            
            # Get frame
            frame = camera.getFrame()
            
            if frame is not None and frame.size > 0:
                # Add focus value overlay
                cv2.putText(frame, f"Focus: {current_focus}", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f"Time: {elapsed:.1f}s", (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
                
                out.write(frame)
                frames_written += 1
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    finally:
        out.release()
        camera.stop()
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"Recording complete!")
        print(f"Duration: {elapsed:.1f}s")
        print(f"Frames written: {frames_written}")
        print(f"Average FPS: {frames_written / elapsed:.1f}")
        print(f"Focus values tested: {focus_index + 1} / {len(focus_values)}")
        print(f"Saved to: {output_file}")
        print("=" * 60)
        
        # Print focus value timeline
        print("\nFocus Timeline:")
        for i, focus in enumerate(focus_values[:focus_index + 1]):
            time_start = i * change_interval
            time_end = (i + 1) * change_interval
            print(f"  {time_start:.0f}s - {time_end:.0f}s: Focus = {focus}")
        print("\nWatch the video to find the best focus value!")


if __name__ == "__main__":
    main()
