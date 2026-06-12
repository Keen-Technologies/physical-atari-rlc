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

#ifndef ROBOTROLLER_H
#define ROBOTROLLER_H

#include "servo.h"
#include "dynamixel_sdk.h"
#include <string>
#include <vector>
#include <mutex>
#include <thread>


class Robotroller
{
private:
    dynamixel::PortHandler* portHandler;
    dynamixel::PacketHandler* packetHandler;
    Servo* fire_servo;
    Servo* up_down_servo;
    Servo* left_right_servo;

    int new_action_to_execute; 
    int last_action_executed;
    std::mutex command_mutex;
    bool robotroller_thread_is_running;
    std::thread robotroller_thread; 

    // Servo position parameters
    int dpad_servo_default;
    int dpad_servo_right;
    int dpad_servo_left;
    int dpad_servo_up;
    int dpad_servo_down;
    int button_servo_default;
    int button_deflection;

    std::vector<Servo*> list_of_servos; 
    struct ServoPositions
    {
        int left_right;
        int up_down;
        int fire;
    };

    ServoPositions getPositionsForAction(int action);
    void sendCommandsToServos();

public:
    Robotroller(std::string device_path, int baud_rate, int position_d_gain, int position_i_gain, int position_p_gain,
                int dpad_servo_default, int dpad_servo_right, int dpad_servo_left, int dpad_servo_up, int dpad_servo_down,
                int button_servo_default, int button_deflection);
    ~Robotroller();
    void setAction(int action);
};


#endif // ROBOTROLLER_H
