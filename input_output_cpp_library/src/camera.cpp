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

#include "../include/camera.h"
#include <chrono>
#include <iostream>

Camera::Camera(int camera_index, int width, int height, int focus_value, int zoom_value,
               int fps_value, int exposure_value, int brightness_value, int contrast_value)
    : camera_index(camera_index), width(width), height(height), focus(focus_value), zoom(zoom_value),
      fps(fps_value), exposure(exposure_value), brightness(brightness_value), contrast(contrast_value),
      is_running(false), new_frame_available(false)
{
    // Open the camera with V4L2 backend
    capture.open(camera_index, cv::CAP_V4L2);


    if (!capture.isOpened())
    {
        std::cerr << "Error: Unable to open camera " << camera_index << std::endl;
        throw std::runtime_error("Failed to open camera");
    }
    
    // Set camera properties
    // Set format to YUYV
    capture.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', 'V'));
    
    // Set FPS
    capture.set(cv::CAP_PROP_FPS, fps);
    
    // Set resolution
    capture.set(cv::CAP_PROP_FRAME_WIDTH, width);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, height);
    
    // Set manual focus mode (disable autofocus)
    capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
    
    // Set focus value
    capture.set(cv::CAP_PROP_FOCUS, focus);
    
    // Set zoom value
    capture.set(cv::CAP_PROP_ZOOM, zoom);
    
    // Set manual exposure mode (disable auto exposure)
    // For V4L2: 1 = Manual Mode, 3 = Aperture Priority Mode (auto)
    // OpenCV uses 0.25 for manual, 0.75 for auto
    capture.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
    
    // Set exposure time (exposure_time_absolute)
    capture.set(cv::CAP_PROP_EXPOSURE, exposure);
    
    // Set brightness
    capture.set(cv::CAP_PROP_BRIGHTNESS, brightness);
    
    // Set contrast
    capture.set(cv::CAP_PROP_CONTRAST, contrast);
    
    // Set buffer size to 1
    capture.set(cv::CAP_PROP_BUFFERSIZE, 1);
    
    // Print actual settings that were applied
    std::cout << "Camera " << camera_index << " initialized successfully" << std::endl;
    std::cout << "Requested settings:" << std::endl;
    std::cout << "  Resolution: " << width << "x" << height << std::endl;
    std::cout << "  FPS: " << fps << std::endl;
    std::cout << "  Focus: " << focus << " (manual)" << std::endl;
    std::cout << "  Zoom: " << zoom << std::endl;
    std::cout << "  Exposure: " << exposure << " (manual)" << std::endl;
    std::cout << "  Brightness: " << brightness << std::endl;
    std::cout << "  Contrast: " << contrast << std::endl;
    
    std::cout << "Actual settings:" << std::endl;
    std::cout << "  Resolution: " << capture.get(cv::CAP_PROP_FRAME_WIDTH) << "x" 
              << capture.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "  FPS: " << capture.get(cv::CAP_PROP_FPS) << std::endl;
    std::cout << "  Format: " << capture.get(cv::CAP_PROP_FOURCC) << std::endl;
    std::cout << "  Buffer size: " << capture.get(cv::CAP_PROP_BUFFERSIZE) << std::endl;
    std::cout << "  Zoom: " << capture.get(cv::CAP_PROP_ZOOM) << std::endl;
    std::cout << "  Focus: " << capture.get(cv::CAP_PROP_FOCUS) << std::endl;
    std::cout << "  Auto Exposure: " << capture.get(cv::CAP_PROP_AUTO_EXPOSURE) << std::endl;
    std::cout << "  Exposure: " << capture.get(cv::CAP_PROP_EXPOSURE) << std::endl;
    std::cout << "  Brightness: " << capture.get(cv::CAP_PROP_BRIGHTNESS) << std::endl;
    std::cout << "  Contrast: " << capture.get(cv::CAP_PROP_CONTRAST) << std::endl;
}

Camera::~Camera()
{
    // Stop the capture thread if still running
    if (is_running)
    {
        stop();
    }
    
    // Release the camera
    if (capture.isOpened())
    {
        capture.release();
    }
}

void Camera::start()
{
    if (is_running)
    {
        std::cerr << "Warning: Camera capture is already running" << std::endl;
        return;
    }
    
    is_running = true;
    capture_thread = std::thread(&Camera::captureLoop, this);
}

void Camera::stop()
{
    if (!is_running)
    {
        return;
    }
    
    is_running = false;
    
    if (capture_thread.joinable())
    {
        capture_thread.join();
    }
}

void Camera::captureLoop()
{
    cv::Mat frame;
    
    while (is_running)
    {
        // Read frame from camera
        if (capture.read(frame))
        {
            // Lock mutex and update current frame
            frame_mutex.lock();
            current_frame = frame.clone();
            new_frame_available = true;
            frame_mutex.unlock();
        }
        
        // Wait 2ms before next attempt (to achieve ~500 FPS attempt rate)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

cv::Mat Camera::getFrame()
{
    // Busy-wait for a new frame to be available
    int timeout_ms = 0;
    const int max_timeout_ms = 10000;  // 10 second timeout
    
    while (is_running)
    {
        frame_mutex.lock();
        if (new_frame_available)
        {
            // Mark frame as read
            new_frame_available = false;
            // Return a copy of the current frame
            cv::Mat frame_copy = current_frame.clone();
            frame_mutex.unlock();
            return frame_copy;
        }
        frame_mutex.unlock();
        
        // Small sleep to avoid burning CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        
        // Check timeout
        timeout_ms += 1;
        if (timeout_ms >= max_timeout_ms)
        {
            std::cerr << "Warning: Frame timeout after " << max_timeout_ms << "ms" << std::endl;
            return cv::Mat();  // Return empty frame on timeout
        }
    }
    
    // Camera stopped, return empty frame
    return cv::Mat();
}

void Camera::setFocus(int focus_value)
{
    focus = focus_value;
    capture.set(cv::CAP_PROP_FOCUS, focus_value);
}

void Camera::setZoom(int zoom_value)
{
    zoom = zoom_value;
    capture.set(cv::CAP_PROP_ZOOM, zoom_value);
}

void Camera::setFps(int fps_value)
{
    fps = fps_value;
    capture.set(cv::CAP_PROP_FPS, fps_value);
}

void Camera::setExposure(int exposure_value)
{
    exposure = exposure_value;
    capture.set(cv::CAP_PROP_EXPOSURE, exposure_value);
}

void Camera::setBrightness(int brightness_value)
{
    brightness = brightness_value;
    capture.set(cv::CAP_PROP_BRIGHTNESS, brightness_value);
}

void Camera::setContrast(int contrast_value)
{
    contrast = contrast_value;
    capture.set(cv::CAP_PROP_CONTRAST, contrast_value);
}

int Camera::getFocus() const
{
    return focus;
}

int Camera::getZoom() const
{
    return zoom;
}

int Camera::getFps() const
{
    return fps;
}

int Camera::getExposure() const
{
    return exposure;
}

int Camera::getBrightness() const
{
    return brightness;
}

int Camera::getContrast() const
{
    return contrast;
}

int Camera::getWidth() const
{
    return width;
}

int Camera::getHeight() const
{
    return height;
}

