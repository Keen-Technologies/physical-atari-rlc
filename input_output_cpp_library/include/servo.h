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

#ifndef SERVO_H
#define SERVO_H

#include <iostream>
#include <string>
#include "dynamixel_sdk.h"

#define ADDR_TORQUE_ENABLE 64
#define ADDR_GOAL_POSITION 116
#define ADDR_PRESENT_POSITION 132

#define ADDR_PRESENT_CURRENT 126
#define PRESENT_CURRENT_BYTE 2

#define ADDR_OPERATING_MODE 11
#define ADDR_POSITION_D_GAIN 80
#define ADDR_POSITION_I_GAIN 82
#define ADDR_POSITION_P_GAIN 84

#define TORQUE_ENABLE 1
#define TORQUE_DISABLE 0


class Servo
{
    int id;
    dynamixel::PacketHandler* pck_handler_;
    dynamixel::PortHandler* portHandler;  // Shared pointer, not owned by Servo

public:
    static bool logDynamixelErrors(
        dynamixel::PacketHandler* packetHandler,
        const int& dxl_comm_result, const uint8_t& dxl_error);

    Servo(int id, dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler);
    int16_t getPresentCurrent();
    int getPresentPosition();
    void setPosition(int position);
    void setPositionDGain(int d_gain);
    void setPositionIGain(int i_gain);
    void setPositionPGain(int p_gain);
    void setModeToPosition();
    void enableTorque();
    void disableTorque();
    ~Servo();
};


#endif //SERVO_H
