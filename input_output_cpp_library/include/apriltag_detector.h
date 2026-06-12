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

#ifndef APRILTAG_DETECTOR_H
#define APRILTAG_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <map>

// Forward declarations for apriltag C library types
extern "C" {
    typedef struct apriltag_family apriltag_family_t;
    typedef struct apriltag_detector apriltag_detector_t;
    typedef struct apriltag_detection apriltag_detection_t;
    typedef struct zarray zarray_t;
}

// Wrapper struct for AprilTag detection results
struct AprilTagDetection {
    int id;
    double center_x;
    double center_y;
    double corners[4][2];  // 4 corners, each with x and y coordinates
    
    AprilTagDetection() : id(-1), center_x(0), center_y(0) {
        for (int i = 0; i < 4; i++) {
            corners[i][0] = 0;
            corners[i][1] = 0;
        }
    }
};

class AprilTagDetector {
private:
    apriltag_detector_t* detector;
    apriltag_family_t* tag_family;
    
public:
    // Constructor: initializes AprilTag detector with tag36h11 family
    // Optimized for speed with single thread and decimation
    AprilTagDetector();
    
    // Destructor: cleans up AprilTag resources
    ~AprilTagDetector();
    
    // Detect all AprilTags in a grayscale image
    // detections: output vector that will be filled with detected tags
    void detectTags(const cv::Mat& gray_frame, std::vector<AprilTagDetection>& detections);
    
    // Compute perspective transformation matrix from corner tags (IDs 0, 1, 2, 3)
    // detections: input vector of all detected tags
    // transform_matrix: output transformation matrix
    // output_width, output_height: dimensions of the output image after transformation
    // Returns true if all 4 corner tags were found and matrix computed successfully
    bool computeTransformMatrix(
        const std::vector<AprilTagDetection>& detections,
        cv::Mat& transform_matrix,
        int output_width,
        int output_height
    );
    
    // Process reward tags to determine if there's a reward
    // detections: input vector of all detected tags
    // last_reward_tag: in/out parameter tracking the last reward state tag (10, 11, or 12)
    // last_change_indicator: in/out parameter tracking the last change indicator (15, 16, or 17)
    // Returns: 0 for no reward, +1 for positive reward, -1 for negative reward
    int processRewardTags(
        const std::vector<AprilTagDetection>& detections,
        int& last_reward_tag,
        int& last_change_indicator
    );
};

#endif // APRILTAG_DETECTOR_H

