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

#include "../include/servo.h"

Servo::~Servo()
{
    std::cout << "Destructor called for servo " << this->id << "\n";
    this->disableTorque();
}

void Servo::setPosition(int position)
{
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->write4ByteTxRx(portHandler, this->id, ADDR_GOAL_POSITION,
                                                        position, &dxl_error);
    if (!Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error))
    {
        std::cout << "Failed to set position of servo " << this->id << "\n";
    }
}

void Servo::setPositionDGain(int d_gain)
{
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->write2ByteTxRx(portHandler, this->id, ADDR_POSITION_D_GAIN,
                                                        d_gain, &dxl_error);
    if (!Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error))
    {
        std::cout << "Failed to set Position D Gain of servo " << this->id << "\n";
    }
}

void Servo::setPositionIGain(int i_gain)
{
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->write2ByteTxRx(portHandler, this->id, ADDR_POSITION_I_GAIN,
                                                        i_gain, &dxl_error);
    if (!Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error))
    {
        std::cout << "Failed to set Position I Gain of servo " << this->id << "\n";
    }
}

void Servo::setPositionPGain(int p_gain)
{
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->write2ByteTxRx(portHandler, this->id, ADDR_POSITION_P_GAIN,
                                                        p_gain, &dxl_error);
    if (!Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error))
    {
        std::cout << "Failed to set Position P Gain of servo " << this->id << "\n";
    }
}

int16_t Servo::getPresentCurrent()
{
    uint16_t current = 0;
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->read2ByteTxRx(portHandler, this->id, ADDR_PRESENT_CURRENT,
                                                       &current, &dxl_error);

    Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error);
    return static_cast<int16_t>(current);
}

int Servo::getPresentPosition()
{
    uint32_t position = 0;
    uint8_t dxl_error = 0;
    auto dxl_comm_result = pck_handler_->read4ByteTxRx(portHandler, this->id, ADDR_PRESENT_POSITION,
                                                       &position, &dxl_error);

    Servo::logDynamixelErrors(pck_handler_, dxl_comm_result, dxl_error);
    return static_cast<int>(position);
}


bool Servo::logDynamixelErrors(
    dynamixel::PacketHandler* packetHandler,
    const int& dxl_comm_result, const uint8_t& dxl_error)
{
    if (dxl_comm_result != COMM_SUCCESS)
    {
        printf("%s\n", packetHandler->getTxRxResult(dxl_comm_result));
        return false;
    }
    else if (dxl_error != 0)
    {
        printf("%s\n", packetHandler->getRxPacketError(dxl_error));
        return false;
    }
    return true;
}


Servo::Servo(int id, dynamixel::PortHandler* portHandler, dynamixel::PacketHandler* packetHandler)
{
    this->portHandler = portHandler;
    this->pck_handler_ = packetHandler;
    this->id = id;
}

void Servo::disableTorque()
{
    uint8_t dxl_error = 0;
    pck_handler_->write1ByteTxRx(portHandler, this->id, ADDR_TORQUE_ENABLE, 0, &dxl_error);
}

void Servo::enableTorque()
{
    uint8_t dxl_error = 0;
    pck_handler_->write1ByteTxRx(portHandler, this->id, ADDR_TORQUE_ENABLE, 1, &dxl_error);
}


void Servo::setModeToPosition()
{
    uint8_t dxl_error = 0;
    pck_handler_->write1ByteTxRx(portHandler, this->id, ADDR_OPERATING_MODE, 3, &dxl_error);
}
