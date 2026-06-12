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

#include "../include/apriltag_detector.h"
#include <iostream>
#include <algorithm>

extern "C" {
    #include <apriltag.h>
    #include <tag36h11.h>
    #include <common/image_u8.h>
}

AprilTagDetector::AprilTagDetector() {
    // Create AprilTag detector
    detector = apriltag_detector_create();
    
    // Configure detector for speed (matching Python config)
    detector->nthreads = 1;  // Single thread to avoid overhead
    detector->quad_decimate = 2.0;  // 2x decimation for speed
    detector->quad_sigma = 0.0;
    detector->refine_edges = 0;  // Skip edge refinement for speed
    detector->decode_sharpening = 0.25;
    
    // Add tag36h11 family (for ArUco compatibility)
    tag_family = tag36h11_create();
    apriltag_detector_add_family(detector, tag_family);
    
    std::cout << "[AprilTagDetector] Initialized with tag36h11 family" << std::endl;
}

AprilTagDetector::~AprilTagDetector() {
    // Clean up resources
    if (detector != nullptr) {
        apriltag_detector_destroy(detector);
    }
    if (tag_family != nullptr) {
        tag36h11_destroy(tag_family);
    }
}

void AprilTagDetector::detectTags(const cv::Mat& gray_frame, std::vector<AprilTagDetection>& detections) {
    detections.clear();
    
    if (gray_frame.empty()) {
        return;
    }
    
    // Convert OpenCV Mat to apriltag image_u8_t (zero-copy wrapper)
    image_u8_t img_header = {
        .width = gray_frame.cols,
        .height = gray_frame.rows,
        .stride = gray_frame.cols,
        .buf = gray_frame.data
    };
    
    // Detect tags
    zarray_t* raw_detections = apriltag_detector_detect(detector, &img_header);
    
    // Convert to our wrapper format
    int num_detections = zarray_size(raw_detections);
    for (int i = 0; i < num_detections; i++) {
        apriltag_detection_t* det;
        zarray_get(raw_detections, i, &det);
        
        AprilTagDetection tag_det;
        tag_det.id = det->id;
        tag_det.center_x = det->c[0];
        tag_det.center_y = det->c[1];
        
        // Copy corner positions
        for (int j = 0; j < 4; j++) {
            tag_det.corners[j][0] = det->p[j][0];
            tag_det.corners[j][1] = det->p[j][1];
        }
        
        detections.push_back(tag_det);
    }
    
    // Clean up detections
    apriltag_detections_destroy(raw_detections);
}

bool AprilTagDetector::computeTransformMatrix(
    const std::vector<AprilTagDetection>& detections,
    cv::Mat& transform_matrix,
    int output_width,
    int output_height
) {
    // Find corner tags (IDs 0, 1, 2, 3)
    std::map<int, AprilTagDetection> corner_tags;
    
    for (const auto& det : detections) {
        if (det.id >= 0 && det.id <= 3) {
            corner_tags[det.id] = det;
        }
    }
    
    // Need all 4 corner tags
    if (corner_tags.size() != 4) {
        return false;
    }
    
    // Helper function to find specific corner of a tag
    auto find_corner = [](const double corners[4][2], const std::string& corner_type) -> cv::Point2f {
        // corners array has 4 points with x, y coordinates
        std::vector<cv::Point2f> pts;
        for (int i = 0; i < 4; i++) {
            pts.push_back(cv::Point2f(corners[i][0], corners[i][1]));
        }
        
        if (corner_type == "top_left") {
            // Minimize x + y
            auto min_it = std::min_element(pts.begin(), pts.end(), 
                [](const cv::Point2f& a, const cv::Point2f& b) {
                    return (a.x + a.y) < (b.x + b.y);
                });
            return *min_it;
        } else if (corner_type == "top_right") {
            // Maximize x - y
            auto max_it = std::max_element(pts.begin(), pts.end(), 
                [](const cv::Point2f& a, const cv::Point2f& b) {
                    return (a.x - a.y) < (b.x - b.y);
                });
            return *max_it;
        } else if (corner_type == "bottom_left") {
            // Maximize y - x
            auto max_it = std::max_element(pts.begin(), pts.end(), 
                [](const cv::Point2f& a, const cv::Point2f& b) {
                    return (a.y - a.x) < (b.y - b.x);
                });
            return *max_it;
        } else if (corner_type == "bottom_right") {
            // Maximize x + y
            auto max_it = std::max_element(pts.begin(), pts.end(), 
                [](const cv::Point2f& a, const cv::Point2f& b) {
                    return (a.x + a.y) < (b.x + b.y);
                });
            return *max_it;
        }
        return cv::Point2f(0, 0);
    };
    
    // Extract the specific corners we need from each tag
    cv::Point2f top_left = find_corner(corner_tags[0].corners, "top_left");
    cv::Point2f top_right = find_corner(corner_tags[1].corners, "top_right");
    cv::Point2f bottom_left = find_corner(corner_tags[2].corners, "bottom_left");
    cv::Point2f bottom_right = find_corner(corner_tags[3].corners, "bottom_right");
    
    // Source points (detected corners in original image)
    std::vector<cv::Point2f> src_points = {
        top_left,
        top_right,
        bottom_left,
        bottom_right
    };
    
    // Destination points (where we want corners to be mapped to)
    std::vector<cv::Point2f> dst_points = {
        cv::Point2f(12, 12),
        cv::Point2f(output_width - 12, 12),
        cv::Point2f(12, output_height - 12),
        cv::Point2f(output_width - 12, output_height - 12)
    };
    
    // Compute perspective transform
    transform_matrix = cv::getPerspectiveTransform(src_points, dst_points);
    
    return true;
}

int AprilTagDetector::processRewardTags(
    const std::vector<AprilTagDetection>& detections,
    int& last_reward_tag,
    int& last_change_indicator
) {
    // Filter reward tags (IDs 10, 11, 12, 15, 16, 17)
    std::vector<AprilTagDetection> reward_tags;
    
    for (const auto& det : detections) {
        if (det.id == 10 || det.id == 11 || det.id == 12 || 
            det.id == 15 || det.id == 16 || det.id == 17) {
            reward_tags.push_back(det);
        }
    }
    
    // Need at least 2 tags (reward state + change indicator)
    if (reward_tags.size() < 2) {
        if (reward_tags.size() == 1) {
            // Update last reward tag if it's a state tag
            int tag_id = reward_tags[0].id;
            if (tag_id == 10 || tag_id == 11 || tag_id == 12) {
                last_reward_tag = tag_id;
            }
        }
        return 0;  // No reward
    }
    
    // Sort tags by vertical position (top to bottom)
    std::sort(reward_tags.begin(), reward_tags.end(), 
        [](const AprilTagDetection& a, const AprilTagDetection& b) {
            return a.center_y < b.center_y;
        });
    
    // First tag (top) is reward state, second tag is change indicator
    int reward_state_tag_id = reward_tags[0].id;
    int change_indicator_tag_id = reward_tags[1].id;
    
    int reward = 0;
    
    // Check if change indicator has changed
    if (change_indicator_tag_id != last_change_indicator) {
        // Change detected - there was a non-zero reward
        if (reward_state_tag_id == 11) {
            reward = 1;  // Positive reward
            std::cout << "[AprilTagDetector] Positive reward detected (+1), change indicator: " 
                      << last_change_indicator << " -> " << change_indicator_tag_id << std::endl;
        } else if (reward_state_tag_id == 12) {
            reward = -1;  // Negative reward
            std::cout << "[AprilTagDetector] Negative reward detected (-1), change indicator: " 
                      << last_change_indicator << " -> " << change_indicator_tag_id << std::endl;
        }
        
        // Update last change indicator
        last_change_indicator = change_indicator_tag_id;
    }
    
    // Update last reward tag
    last_reward_tag = reward_state_tag_id;
    
    return reward;
}

