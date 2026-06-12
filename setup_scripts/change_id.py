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
Script to change the ID of a Dynamixel XC330 servo.
"""

import argparse
from dynamixel_sdk import PortHandler, PacketHandler

# XC330 Control Table Addresses
ADDR_ID = 7
PROTOCOL_VERSION = 2.0


def main():
    parser = argparse.ArgumentParser(description='Change ID of Dynamixel XC330 servo')
    parser.add_argument('--path', type=str, required=True,
                        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/serial/by-id/...)')
    parser.add_argument('--current_id', type=int, required=True,
                        help='Current ID of the servo')
    parser.add_argument('--new_id', type=int, required=True,
                        help='New ID to set for the servo')
    parser.add_argument('--baud_rate', type=int, required=True,
                        help='Baud rate (e.g., 1000000)')
    
    args = parser.parse_args()
    
    # Validate ID range (Dynamixel IDs are typically 0-253)
    if args.current_id < 0 or args.current_id > 253:
        print(f"Error: Current ID must be between 0 and 253, got {args.current_id}")
        return
    
    if args.new_id < 0 or args.new_id > 253:
        print(f"Error: New ID must be between 0 and 253, got {args.new_id}")
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
    
    # Write new ID to the servo
    print(f"Changing servo ID from {args.current_id} to {args.new_id}...")
    result, error = packet_handler.write1ByteTxRx(port_handler, args.current_id, ADDR_ID, args.new_id)
    
    if result != 0:
        print(f"Communication error: {packet_handler.getTxRxResult(result)}")
        port_handler.closePort()
        return
    
    if error != 0:
        print(f"Error from servo: {packet_handler.getRxPacketError(error)}")
        port_handler.closePort()
        return
    
    print(f"Successfully changed servo ID from {args.current_id} to {args.new_id}")
    print("Note: The servo will use the new ID immediately. Power cycle may be required for some changes.")
    
    # Close port
    port_handler.closePort()
    print("Port closed")


if __name__ == "__main__":
    main()
