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

#include "../include/physical_atari_env.h"
#include <iostream>
#include <thread>
#include <chrono>

PhysicalAtariEnv::PhysicalAtariEnv(
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
) : camera(camera_index, width, height, focus_value, zoom_value, fps_value, exposure_value, brightness_value, contrast_value),
    robotroller(serial_port, baud_rate, position_d_gain, position_i_gain, position_p_gain,
                dpad_servo_default, dpad_servo_right, dpad_servo_left, dpad_servo_up, dpad_servo_down,
                button_servo_default, button_deflection),
    detector(),
    output_width(1280),
    output_height(720),
    game_crop_x(193),
    game_crop_y(44),
    game_crop_w(930),
    game_crop_h(647),
    max_frames_without_reward(18000),
    pending_reward(0),
    last_reward(0),
    terminated(false),
    truncated(false),
    frame_count(0),
    frames_without_reward(0),
    last_reward_tag(10),
    last_change_indicator(15)
{
    std::cout << "[PhysicalAtariEnv] Initializing environment for game: " << game << std::endl;
    
    // Start camera
    camera.start();
    
    // Wait for camera to warm up
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    // Initialize observation with correct size (210x160x3 in RGB format)
    current_observation = cv::Mat::zeros(210, 160, CV_8UC3);
    
    // Capture initial frame to set up transformation
    captureAndProcessFrame();
    
    std::cout << "[PhysicalAtariEnv] Environment initialized successfully" << std::endl;
}

PhysicalAtariEnv::~PhysicalAtariEnv() {
    camera.stop();
}

void PhysicalAtariEnv::captureAndProcessFrame() {
    // Capture frame from camera
    cv::Mat frame = camera.getFrame();
    
    if (frame.empty()) {
        std::cerr << "[PhysicalAtariEnv] Warning: Empty frame captured" << std::endl;
        return;
    }
    
    frame_count++;
    
    // Convert to grayscale for AprilTag detection
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    
    // Detect all AprilTags
    std::vector<AprilTagDetection> all_tags;
    detector.detectTags(gray, all_tags);
    
    // Try to compute/update transformation matrix from corner tags
    cv::Mat new_transform;
    if (detector.computeTransformMatrix(all_tags, new_transform, output_width, output_height)) {
        state_mutex.lock();
        transform_matrix = new_transform;
        state_mutex.unlock();
    }
    
    // Apply transformation if we have a valid matrix
    state_mutex.lock();
    bool have_transform = !transform_matrix.empty();
    cv::Mat transform_copy = transform_matrix.clone();
    state_mutex.unlock();
    
    if (have_transform) {
        // Apply perspective transformation
        cv::Mat transformed_frame;
        cv::warpPerspective(frame, transformed_frame, transform_copy, 
                           cv::Size(output_width, output_height));
        
        // Crop game area
        cv::Rect game_roi(game_crop_x, game_crop_y, game_crop_w, game_crop_h);
        cv::Mat game_crop = transformed_frame(game_roi);
        
        // Resize to ALE format (160x210)
        cv::Mat resized;
        cv::resize(game_crop, resized, cv::Size(160, 210));
        
        // Convert BGR to RGB
        cv::Mat rgb_frame;
        cv::cvtColor(resized, rgb_frame, cv::COLOR_BGR2RGB);
        
        // Update current observation
        state_mutex.lock();
        current_observation = rgb_frame.clone();
        state_mutex.unlock();
        
        // Process reward tags
        int reward = detector.processRewardTags(all_tags, last_reward_tag, last_change_indicator);
        
        state_mutex.lock();
        pending_reward = reward;
        state_mutex.unlock();
        
        if (reward != 0) {
            std::cout << "[PhysicalAtariEnv] Frame " << frame_count << ": Reward = " << reward << std::endl;
        }
    } else {
        // No transformation matrix yet, keep black frame
        std::cout << "[PhysicalAtariEnv] Warning: No transformation matrix available yet" << std::endl;
    }
}

void PhysicalAtariEnv::step(int action_index) {
    int physical_action = (action_index >= 0 && action_index <= 17) ? action_index : 0;
    robotroller.setAction(physical_action);
    
    // Capture and process frame
    captureAndProcessFrame();
    
    // Update frames without reward counter
    state_mutex.lock();
    last_reward = pending_reward;  // Save reward for getReward()
    pending_reward = 0;  // Reset pending reward after reading
    state_mutex.unlock();
    
    if (last_reward != 0) {
        frames_without_reward = 0;
    } else {
        frames_without_reward++;
    }
    
    // Check termination conditions
    terminated = false;  // Can't easily detect game over in physical env
    truncated = (frames_without_reward >= max_frames_without_reward);
}

const cv::Mat& PhysicalAtariEnv::getObservation() const {
    return current_observation;
}

int PhysicalAtariEnv::getReward() const {
    state_mutex.lock();
    int reward = last_reward;
    state_mutex.unlock();
    return reward;
}

bool PhysicalAtariEnv::getTerminated() const {
    return terminated;
}

bool PhysicalAtariEnv::getTruncated() const {
    return truncated;
}

void PhysicalAtariEnv::reset() {
    std::cout << "[PhysicalAtariEnv] Resetting environment" << std::endl;
    
    // Reset state
    state_mutex.lock();
    pending_reward = 0;
    last_reward = 0;
    terminated = false;
    truncated = false;
    frames_without_reward = 0;
    last_reward_tag = 10;  // Expect tag 10 at start
    last_change_indicator = 15;  // Expect tag 15 at start
    state_mutex.unlock();
    
    // Don't reset transformation matrix - it persists across episodes
    
    // Capture initial frame
    captureAndProcessFrame();
}

int PhysicalAtariEnv::getNumActions() const {
    return 18;  // Full action set (0-17)
}

int PhysicalAtariEnv::getFrameCount() const {
    return frame_count;
}

