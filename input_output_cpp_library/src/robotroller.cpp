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

#include "../include/robotroller.h"
#include "../include/servo.h"
#include <thread>
#include <chrono>
#include <iostream>
#include <cstdlib>

#define PROTOCOL_VERSION 2.0

Robotroller::ServoPositions Robotroller::getPositionsForAction(int action)
{
    ServoPositions positions[18] = {
        {dpad_servo_default, dpad_servo_default, button_servo_default}, // 0: NOOP
        {dpad_servo_default, dpad_servo_default, button_deflection}, // 1: FIRE
        {dpad_servo_default, dpad_servo_up, button_servo_default}, // 2: UP
        {dpad_servo_right, dpad_servo_default, button_servo_default}, // 3: RIGHT
        {dpad_servo_left, dpad_servo_default, button_servo_default}, // 4: LEFT
        {dpad_servo_default, dpad_servo_down, button_servo_default}, // 5: DOWN
        {dpad_servo_right, dpad_servo_up, button_servo_default}, // 6: UPRIGHT
        {dpad_servo_left, dpad_servo_up, button_servo_default}, // 7: UPLEFT
        {dpad_servo_right, dpad_servo_down, button_servo_default}, // 8: DOWNRIGHT
        {dpad_servo_left, dpad_servo_down, button_servo_default}, // 9: DOWNLEFT
        {dpad_servo_default, dpad_servo_up, button_deflection}, // 10: UPFIRE
        {dpad_servo_right, dpad_servo_default, button_deflection}, // 11: RIGHTFIRE
        {dpad_servo_left, dpad_servo_default, button_deflection}, // 12: LEFTFIRE
        {dpad_servo_default, dpad_servo_down, button_deflection}, // 13: DOWNFIRE
        {dpad_servo_right, dpad_servo_up, button_deflection}, // 14: UPRIGHTFIRE
        {dpad_servo_left, dpad_servo_up, button_deflection}, // 15: UPLEFTFIRE
        {dpad_servo_right, dpad_servo_down, button_deflection}, // 16: DOWNRIGHTFIRE
        {dpad_servo_left, dpad_servo_down, button_deflection} // 17: DOWNLEFTFIRE
    };

    return positions[action];
}


Robotroller::Robotroller(std::string device_path, int baud_rate, int position_d_gain, int position_i_gain, int position_p_gain,
                         int dpad_servo_default, int dpad_servo_right, int dpad_servo_left, int dpad_servo_up, int dpad_servo_down,
                         int button_servo_default, int button_deflection)
    : portHandler(nullptr),
      packetHandler(nullptr),
      fire_servo(nullptr),
      left_right_servo(nullptr),
      up_down_servo(nullptr),
      dpad_servo_default(dpad_servo_default),
      dpad_servo_right(dpad_servo_right),
      dpad_servo_left(dpad_servo_left),
      dpad_servo_up(dpad_servo_up),
      dpad_servo_down(dpad_servo_down),
      button_servo_default(button_servo_default),
      button_deflection(button_deflection),
      robotroller_thread_is_running(true),
      new_action_to_execute(-1), 
      last_action_executed(0)
{
    // Create and initialize port handler
    std::cout << "[Robotroller] Attempting to open serial port: " << device_path << std::endl;
    portHandler = dynamixel::PortHandler::getPortHandler(device_path.c_str());

    if (!portHandler->openPort())
    {
        std::cerr << "[Robotroller] Failed to open the serial port to dynamixel motors!" << std::endl;
        exit(1);
    }
    std::cout << "[Robotroller] Succeeded to open the port!" << std::endl;
    
    // Set baud rate
    if (!portHandler->setBaudRate(baud_rate))
    {
        std::cerr << "[Robotroller] Failed to set baud rate to " << baud_rate << std::endl;
        portHandler->closePort();
        exit(1);
    }
    std::cout << "[Robotroller] Baud rate set to " << baud_rate << std::endl;
    
    // Clear any garbage in serial buffers
    portHandler->clearPort();
    
    // Create packet handler
    packetHandler = dynamixel::PacketHandler::getPacketHandler(PROTOCOL_VERSION);
    
    // Initialize servos with shared port and packet handlers
    fire_servo = new Servo(50, portHandler, packetHandler);
    left_right_servo = new Servo(51, portHandler, packetHandler);
    up_down_servo = new Servo(52, portHandler, packetHandler);
    
    list_of_servos.push_back(fire_servo);
    list_of_servos.push_back(left_right_servo);
    list_of_servos.push_back(up_down_servo);
    fire_servo->disableTorque();
    fire_servo->setModeToPosition();
    if (position_d_gain > 0)
    {
        fire_servo->setPositionDGain(position_d_gain);
    }
    if (position_i_gain > 0)
    {
        fire_servo->setPositionIGain(position_i_gain);
    }
    if (position_p_gain > 0)
    {
        fire_servo->setPositionPGain(position_p_gain);
    }
    fire_servo->enableTorque();

    up_down_servo->disableTorque();
    up_down_servo->setModeToPosition();
    if (position_d_gain > 0)
    {
        up_down_servo->setPositionDGain(position_d_gain);
    }
    if (position_i_gain > 0)
    {
        up_down_servo->setPositionIGain(position_i_gain);
    }
    if (position_p_gain > 0)
    {
        up_down_servo->setPositionPGain(position_p_gain);
    }
    up_down_servo->enableTorque();

    left_right_servo->disableTorque();
    left_right_servo->setModeToPosition();
    if (position_d_gain > 0)
    {
        left_right_servo->setPositionDGain(position_d_gain);
    }
    if (position_i_gain > 0)
    {
        left_right_servo->setPositionIGain(position_i_gain);
    }
    if (position_p_gain > 0)
    {
        left_right_servo->setPositionPGain(position_p_gain);
    }
    left_right_servo->enableTorque();
    
    // Start the command thread
    robotroller_thread = std::thread(&Robotroller::sendCommandsToServos, this);
}

Robotroller::~Robotroller()
{
    // Stop the thread
    robotroller_thread_is_running = false;
    
    // Wait for thread to finish
    if (robotroller_thread.joinable())
    {
        robotroller_thread.join();
    }
    
    // Delete servos
    fire_servo->disableTorque();
    left_right_servo->disableTorque();
    up_down_servo->disableTorque();
    delete fire_servo;
    delete left_right_servo;
    delete up_down_servo;
    
    // Close port when robotroller is destroyed
    if (portHandler != nullptr)
    {
        portHandler->closePort();
    }
}

void Robotroller::sendCommandsToServos()
{

    while(true){
        command_mutex.lock();
        if(new_action_to_execute != -1)
        {
            int temp_action = new_action_to_execute;
            command_mutex.unlock();
            ServoPositions pos = getPositionsForAction(temp_action);
            left_right_servo->setPosition(pos.left_right);
            up_down_servo->setPosition(pos.up_down);
            fire_servo->setPosition(pos.fire);

            command_mutex.lock();
            last_action_executed = new_action_to_execute;
            new_action_to_execute = -1;
            
        }
        command_mutex.unlock();


        for (auto servo : list_of_servos) {
            int16_t cur = servo->getPresentCurrent();
            if (std::abs(cur) > 1200) {
                servo->setPosition(servo->getPresentPosition());
                std::cout << "Disabling torque\n";
                servo->disableTorque();
                std::cout << "Enabling torque\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                servo->enableTorque();
                std::cout << "Overcurrent detected\n";
            }
        }


        if(!robotroller_thread_is_running)
            break; 
    }
}

void Robotroller::setAction(int action)
{
    // Skip if same action
    command_mutex.lock();
    if (action == new_action_to_execute || (new_action_to_execute == -1 && action == last_action_executed))
    {
        command_mutex.unlock();
        return;
    }
    if (new_action_to_execute != -1)
    {
        // Another command is in progress, skip this one
        command_mutex.unlock();
        return;
    }
    new_action_to_execute = action;
    command_mutex.unlock();
}
