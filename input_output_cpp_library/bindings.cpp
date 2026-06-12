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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "include/camera.h"
#include "include/robotroller.h"
#include "include/physical_atari_env.h"

namespace py = pybind11;

// Wrapper for Camera::getFrame() that returns a numpy array backed by a cloned frame
py::array_t<uint8_t> Camera_getFrame_wrapper(Camera* camera) {
    cv::Mat frame = camera->getFrame();
    
    if (frame.empty()) {
        return py::array_t<uint8_t>();
    }
    
    // getFrame() returns a copy; wrap it in a capsule so Python owns the buffer
    py::capsule mat_capsule(new cv::Mat(frame), [](void *mat) {
        delete reinterpret_cast<cv::Mat*>(mat);
    });
    
    return py::array_t<uint8_t>(
        {frame.rows, frame.cols, frame.channels()},  // shape
        {frame.step[0], frame.step[1], sizeof(uint8_t)},  // strides
        frame.data,  // data pointer
        mat_capsule  // capsule to manage lifetime
    );
}

// Wrapper for PhysicalAtariEnv::getObservation() that returns a numpy array with zero-copy
py::array_t<uint8_t> PhysicalAtariEnv_getObservation_wrapper(PhysicalAtariEnv* env) {
    const cv::Mat& frame = env->getObservation();
    
    if (frame.empty()) {
        return py::array_t<uint8_t>();
    }
    
    // Create numpy array that views the cv::Mat data (zero-copy)
    // Note: We need to clone the frame since getObservation() returns a const reference
    // and the underlying data might change on the next step
    cv::Mat frame_copy = frame.clone();
    
    py::capsule mat_capsule(new cv::Mat(frame_copy), [](void *mat) {
        delete reinterpret_cast<cv::Mat*>(mat);
    });
    
    return py::array_t<uint8_t>(
        {frame_copy.rows, frame_copy.cols, frame_copy.channels()},  // shape
        {frame_copy.step[0], frame_copy.step[1], sizeof(uint8_t)},  // strides
        frame_copy.data,  // data pointer
        mat_capsule  // capsule to manage lifetime
    );
}

PYBIND11_MODULE(robotroller, m) {
    m.doc() = "Python bindings for Robotroller, Camera, and PhysicalAtariEnv";
    
    // Camera class bindings
    py::class_<Camera>(m, "Camera")
        .def(py::init<int, int, int, int, int, int, int, int, int>(),
             py::arg("camera_index"),
             py::arg("width") = 1280,
             py::arg("height") = 720,
             py::arg("focus_value") = 370,
             py::arg("zoom_value") = 100,
             py::arg("fps_value") = 30,
             py::arg("exposure_value") = 20,
             py::arg("brightness_value") = 128,
             py::arg("contrast_value") = 128,
             "Initialize camera with specified parameters")
        .def("start", &Camera::start,
             "Start the camera capture thread")
        .def("stop", &Camera::stop,
             "Stop the camera capture thread")
        .def("getFrame", &Camera_getFrame_wrapper,
             "Get the next frame as a numpy array")
        .def("setFocus", &Camera::setFocus,
             py::arg("focus_value"),
             "Set the camera focus value")
        .def("setZoom", &Camera::setZoom,
             py::arg("zoom_value"),
             "Set the camera zoom value")
        .def("setFps", &Camera::setFps,
             py::arg("fps_value"),
             "Set the camera FPS")
        .def("setExposure", &Camera::setExposure,
             py::arg("exposure_value"),
             "Set the camera exposure value")
        .def("setBrightness", &Camera::setBrightness,
             py::arg("brightness_value"),
             "Set the camera brightness value")
        .def("setContrast", &Camera::setContrast,
             py::arg("contrast_value"),
             "Set the camera contrast value")
        .def("getFocus", &Camera::getFocus,
             "Get the camera focus value")
        .def("getZoom", &Camera::getZoom,
             "Get the camera zoom value")
        .def("getFps", &Camera::getFps,
             "Get the camera FPS")
        .def("getExposure", &Camera::getExposure,
             "Get the camera exposure value")
        .def("getBrightness", &Camera::getBrightness,
             "Get the camera brightness value")
        .def("getContrast", &Camera::getContrast,
             "Get the camera contrast value")
        .def("getWidth", &Camera::getWidth,
             "Get the camera width")
        .def("getHeight", &Camera::getHeight,
             "Get the camera height");
    
    // Robotroller class bindings
    py::class_<Robotroller>(m, "Robotroller")
        .def(py::init<std::string, int, int, int, int, int, int, int, int, int, int, int>(),
             py::arg("device_path"),
             py::arg("baud_rate") = 1000000,
             py::arg("position_d_gain") = 0,
             py::arg("position_i_gain") = 0,
             py::arg("position_p_gain") = 0,
             py::arg("dpad_servo_default"),
             py::arg("dpad_servo_right"),
             py::arg("dpad_servo_left"),
             py::arg("dpad_servo_up"),
             py::arg("dpad_servo_down"),
             py::arg("button_servo_default"),
             py::arg("button_deflection"),
             "Initialize Robotroller with device path, baud rate, position PID gains, and servo positions")
        .def("setAction", &Robotroller::setAction,
             py::arg("action"),
             "Set the action (0-17)");
    
    // PhysicalAtariEnv class bindings
    py::class_<PhysicalAtariEnv>(m, "PhysicalAtariEnv")
        .def(py::init<std::string, int, int, int, int, int, int, int, int, int, int, std::string, int, int, int, int, int, int, int, int, int, int, int>(),
             py::arg("game"),
             py::arg("seed"),
             py::arg("camera_index"),
             py::arg("width"),
             py::arg("height"),
             py::arg("focus_value"),
             py::arg("zoom_value"),
             py::arg("fps_value"),
             py::arg("exposure_value"),
             py::arg("brightness_value"),
             py::arg("contrast_value"),
             py::arg("serial_port"),
             py::arg("position_d_gain"),
             py::arg("position_i_gain"),
             py::arg("position_p_gain"),
             py::arg("baud_rate"),
             py::arg("dpad_servo_default"),
             py::arg("dpad_servo_right"),
             py::arg("dpad_servo_left"),
             py::arg("dpad_servo_up"),
             py::arg("dpad_servo_down"),
             py::arg("button_servo_default"),
             py::arg("button_deflection"),
             "Initialize PhysicalAtariEnv with game name and hardware parameters")
        .def("step", &PhysicalAtariEnv::step,
             py::arg("action_index"),
             "Execute one step in the environment with the given action")
        .def("getObservation", &PhysicalAtariEnv_getObservation_wrapper,
             "Get the current RGB observation as a numpy array (210x160x3)")
        .def("getReward", &PhysicalAtariEnv::getReward,
             "Get the reward from the last step")
        .def("getTerminated", &PhysicalAtariEnv::getTerminated,
             "Check if episode ended (game over)")
        .def("getTruncated", &PhysicalAtariEnv::getTruncated,
             "Check if episode was truncated (max frames without reward)")
        .def("reset", &PhysicalAtariEnv::reset,
             "Reset the environment for a new episode")
        .def("getNumActions", &PhysicalAtariEnv::getNumActions,
             "Get the number of valid actions")
        .def("getFrameCount", &PhysicalAtariEnv::getFrameCount,
             "Get the current frame count");
}

