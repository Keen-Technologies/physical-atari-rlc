#!/usr/bin/env python3
# Copyright 2026 Keen Technologies, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to change the position of a Dynamixel XC330 servo.
Sets the servo to current-based position control mode, sets the current limit,
and then sets the goal position.
"""

import argparse
import time
from dynamixel_sdk import PortHandler, PacketHandler

# XC330 Control Table Addresses
ADDR_OPERATING_MODE = 11
ADDR_GOAL_CURRENT = 102
ADDR_GOAL_POSITION = 116
ADDR_TORQUE_ENABLE = 64
PROTOCOL_VERSION = 2.0

# Operating Mode Values
MODE_CURRENT_BASED_POSITION = 5

# Torque Values
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0


def main():
    parser = argparse.ArgumentParser(description='Change position of Dynamixel XC330 servo')
    parser.add_argument('--path', type=str, required=True,
                        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/serial/by-id/...)')
    parser.add_argument('--id', type=int, required=True,
                        help='ID of the servo')
    parser.add_argument('--position', type=int, required=True,
                        help='Goal position to set')
    parser.add_argument('--baud_rate', type=int, required=True,
                        help='Baud rate (e.g., 1000000)')
    parser.add_argument('--current_limit', type=int, default=275,
                        help='Current limit (default: 275)')
    
    args = parser.parse_args()
    
    # Validate ID range (Dynamixel IDs are typically 0-253)
    if args.id < 0 or args.id > 253:
        print(f"Error: ID must be between 0 and 253, got {args.id}")
        return
    
    # Initialize PortHandler
    port_handler = PortHandler(args.path)
    
    # Initialize PacketHandler
    packet_handler = PacketHandler(PROTOCOL_VERSION)
    
    # Open port
    if not port_handler.openPort():
        print(f"Failed to open port: {args.path}")
        return
    
    print(f"Successfully opened port: {args.path}")
    
    # Set baud rate
    if not port_handler.setBaudRate(args.baud_rate):
        print(f"Failed to set baud rate to {args.baud_rate}")
        port_handler.closePort()
        return
    
    print(f"Baud rate set to {args.baud_rate}")
    
    try:
        # Set operating mode to current-based position control
        print(f"Setting servo {args.id} to current-based position control mode...")
        result, error = packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_OPERATING_MODE, MODE_CURRENT_BASED_POSITION)
        
        if result != 0:
            print(f"Communication error setting operating mode: {packet_handler.getTxRxResult(result)}")
            port_handler.closePort()
            return
        
        if error != 0:
            print(f"Error from servo setting operating mode: {packet_handler.getRxPacketError(error)}")
            port_handler.closePort()
            return
        
        print(f"Successfully set operating mode to current-based position control")
        
        # Set goal current (current limit)
        print(f"Setting goal current (current limit) to {args.current_limit}...")
        result, error = packet_handler.write2ByteTxRx(port_handler, args.id, ADDR_GOAL_CURRENT, args.current_limit)
        
        if result != 0:
            print(f"Communication error setting goal current: {packet_handler.getTxRxResult(result)}")
            port_handler.closePort()
            return
        
        if error != 0:
            print(f"Error from servo setting goal current: {packet_handler.getRxPacketError(error)}")
            port_handler.closePort()
            return
        
        print(f"Successfully set goal current to {args.current_limit}")
        
        # Enable torque
        print(f"Enabling torque for servo {args.id}...")
        result, error = packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE)
        
        if result != 0:
            print(f"Communication error enabling torque: {packet_handler.getTxRxResult(result)}")
            port_handler.closePort()
            return
        
        if error != 0:
            print(f"Error from servo enabling torque: {packet_handler.getRxPacketError(error)}")
            port_handler.closePort()
            return
        
        print(f"Successfully enabled torque")
        
        # Set goal position
        print(f"Setting goal position to {args.position}...")
        result, error = packet_handler.write4ByteTxRx(port_handler, args.id, ADDR_GOAL_POSITION, args.position)
        
        if result != 0:
            print(f"Communication error setting goal position: {packet_handler.getTxRxResult(result)}")
            port_handler.closePort()
            return
        
        if error != 0:
            print(f"Error from servo setting goal position: {packet_handler.getRxPacketError(error)}")
            port_handler.closePort()
            return
        
        print(f"Successfully set goal position to {args.position}")
        
        # Wait 5 seconds
        print("Waiting 5 seconds...")
        time.sleep(5)
        
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C)")
    
    finally:
        # Disable torque
        print(f"Disabling torque for servo {args.id}...")
        result, error = packet_handler.write1ByteTxRx(port_handler, args.id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
        
        if result != 0:
            print(f"Communication error disabling torque: {packet_handler.getTxRxResult(result)}")
        elif error != 0:
            print(f"Error from servo disabling torque: {packet_handler.getRxPacketError(error)}")
        else:
            print(f"Successfully disabled torque")
        
        # Close port
        port_handler.closePort()
        print("Port closed")


if __name__ == "__main__":
    main()

