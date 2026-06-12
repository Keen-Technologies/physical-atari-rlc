// Copyright 2026 Keen Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef PHYSICAL_ATARI_ENV_H
#define PHYSICAL_ATARI_ENV_H

#include "camera.h"
#include "robotroller.h"
#include "apriltag_detector.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <mutex>

class PhysicalAtariEnv {
private:
    // Components
    Camera camera;
    Robotroller robotroller;
    AprilTagDetector detector;
    
    // Configuration
    int output_width;
    int output_height;
    int game_crop_x;
    int game_crop_y;
    int game_crop_w;
    int game_crop_h;
    int max_frames_without_reward;
    
    // State
    cv::Mat transform_matrix;
    cv::Mat current_observation;
    int pending_reward;
    int last_reward;  // Reward from the last step
    bool terminated;
    bool truncated;
    int frame_count;
    int frames_without_reward;
    
    // Reward tracking
    int last_reward_tag;
    int last_change_indicator;
    
    // Mutex for thread safety (mutable so it can be used in const methods)
    mutable std::mutex state_mutex;
    
    // Internal methods
    void captureAndProcessFrame();
    
public:
    // Constructor
    // game: Name of the Atari game (e.g., "pong", "breakout")
    // seed: Random seed for reproducibility (not used in physical env, kept for compatibility)
    // camera_index: Camera device index (e.g., 0 for /dev/video0)
    // width, height: Camera resolution
    // focus_value: Camera focus value
    // zoom_value: Camera zoom value
    // fps_value: Camera FPS
    // exposure_value: Camera exposure
    // brightness_value: Camera brightness
    // contrast_value: Camera contrast
    // serial_port: Serial port for Robotroller (e.g., "/dev/ttyUSB0")
    // position_d_gain: Position D gain for servos
    // position_i_gain: Position I gain for servos
    // position_p_gain: Position P gain for servos
    // baud_rate: Baud rate for serial communication
    // dpad_servo_default: Default position for dpad servos
    // dpad_servo_right: Right position for dpad servo
    // dpad_servo_left: Left position for dpad servo
    // dpad_servo_up: Up position for dpad servo
    // dpad_servo_down: Down position for dpad servo
    // button_servo_default: Default (unfire) position for button servo
    // button_deflection: Fire position for button servo
    PhysicalAtariEnv(
        const std::string& game,
        int seed,
        int camera_index,
        int width,
        int height,
        int focus_value,
        int zoom_value,
        int fps_value,
        int exposure_value,
        int brightness_value,
        int contrast_value,
        const std::string& serial_port,
        int position_d_gain,
        int position_i_gain,
        int position_p_gain,
        int baud_rate,
        int dpad_servo_default,
        int dpad_servo_right,
        int dpad_servo_left,
        int dpad_servo_up,
        int dpad_servo_down,
        int button_servo_default,
        int button_deflection
    );
    
    // Destructor
    ~PhysicalAtariEnv();
    
    // Execute one step in the environment
    // action_index: Action index (0-17 for full action set)
    void step(int action_index);
    
    // Get the current RGB observation (160x210x3)
    // Returns a reference to avoid copying
    const cv::Mat& getObservation() const;
    
    // Get the reward from the last step
    int getReward() const;
    
    // Check if episode ended (game over)
    bool getTerminated() const;
    
    // Check if episode was truncated (max frames without reward)
    bool getTruncated() const;
    
    // Reset the environment for a new episode
    void reset();
    
    // Get the number of valid actions
    int getNumActions() const;
    
    // Get frame count
    int getFrameCount() const;
};

#endif // PHYSICAL_ATARI_ENV_H

