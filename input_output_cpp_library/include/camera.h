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

#ifndef CAMERA_H
#define CAMERA_H

#include <opencv2/opencv.hpp>
#include <mutex>
#include <thread>
#include <atomic>

class Camera
{
private:
    cv::VideoCapture capture;
    cv::Mat current_frame;
    std::mutex frame_mutex;
    bool new_frame_available;
    std::thread capture_thread;
    std::atomic<bool> is_running;
    
    int camera_index;
    int width;
    int height;
    int focus;
    int zoom;
    int fps;
    int exposure;
    int brightness;
    int contrast;
    
    void captureLoop();

public:
    // Constructor with default values
    Camera(int camera_index, 
           int width = 1280, 
           int height = 720, 
           int focus_value = 370, 
           int zoom_value = 100,
           int fps_value = 30,
           int exposure_value = 20,
           int brightness_value = 128,
           int contrast_value = 128);
    ~Camera();
    
    void start();
    void stop();
    cv::Mat getFrame();
    
    // Setters
    void setFocus(int focus_value);
    void setZoom(int zoom_value);
    void setFps(int fps_value);
    void setExposure(int exposure_value);
    void setBrightness(int brightness_value);
    void setContrast(int contrast_value);
    
    // Getters
    int getFocus() const;
    int getZoom() const;
    int getFps() const;
    int getExposure() const;
    int getBrightness() const;
    int getContrast() const;
    int getWidth() const;
    int getHeight() const;
};

#endif // CAMERA_H

