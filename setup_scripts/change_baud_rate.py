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
Script to change the baud rate of a Dynamixel XC330 servo.
"""

import argparse
from dynamixel_sdk import PortHandler, PacketHandler

# XC330 Control Table Addresses
ADDR_BAUD_RATE = 8
PROTOCOL_VERSION = 2.0

# XC330 baud-rate register values (control table address 8)
BAUD_RATE_TO_REGISTER = {
    9600: 0,
    57600: 1,
    115200: 2,
    200000: 3,
    250000: 4,
    400000: 5,
    500000: 6,
    1000000: 7,
    2000000: 8,
    3000000: 9,
    4000000: 10,
}


def baud_rate_to_register(baud_rate):
    if baud_rate not in BAUD_RATE_TO_REGISTER:
        supported = ", ".join(str(b) for b in sorted(BAUD_RATE_TO_REGISTER))
        raise ValueError(f"Unsupported baud rate {baud_rate}. Supported values: {supported}")
    return BAUD_RATE_TO_REGISTER[baud_rate]


def main():
    parser = argparse.ArgumentParser(description='Change baud rate of Dynamixel XC330 servo')
    parser.add_argument('--path', type=str, required=True,
                        help='Serial port path (e.g., /dev/ttyUSB0 or /dev/serial/by-id/...)')
    parser.add_argument('--id', type=int, required=True,
                        help='ID of the servo')
    parser.add_argument('--current_baud_rate', type=int, default=57600,
                        help='Current baud rate used by the servo (default: 57600, factory setting)')
    parser.add_argument('--new_baud_rate', type=int, required=True,
                        help='New baud rate to set on the servo (e.g., 1000000)')

    args = parser.parse_args()

    if args.id < 0 or args.id > 253:
        print(f"Error: Servo ID must be between 0 and 253, got {args.id}")
        return

    try:
        register_value = baud_rate_to_register(args.new_baud_rate)
    except ValueError as exc:
        print(f"Error: {exc}")
        return

    port_handler = PortHandler(args.path)
    packet_handler = PacketHandler(PROTOCOL_VERSION)

    if not port_handler.openPort():
        print(f"Failed to open port: {args.path}")
        return

    print(f"Successfully opened port: {args.path}")

    if not port_handler.setBaudRate(args.current_baud_rate):
        print(f"Failed to set port baud rate to {args.current_baud_rate}")
        port_handler.closePort()
        return

    print(f"Port baud rate set to {args.current_baud_rate}")

    print(f"Changing servo {args.id} baud rate to {args.new_baud_rate} (register value {register_value})...")
    result, error = packet_handler.write1ByteTxRx(
        port_handler, args.id, ADDR_BAUD_RATE, register_value)

    if result != 0:
        print(f"Communication error: {packet_handler.getTxRxResult(result)}")
        port_handler.closePort()
        return

    if error != 0:
        print(f"Error from servo: {packet_handler.getRxPacketError(error)}")
        port_handler.closePort()
        return

    print(f"Successfully wrote new baud rate to servo {args.id}")

    if not port_handler.setBaudRate(args.new_baud_rate):
        print(f"Failed to set port baud rate to {args.new_baud_rate} for verification")
        port_handler.closePort()
        return

    print(f"Port baud rate set to {args.new_baud_rate}")

    model_number, result, error = packet_handler.ping(port_handler, args.id)
    if result != 0:
        print(f"Warning: Could not ping servo at new baud rate: {packet_handler.getTxRxResult(result)}")
    elif error != 0:
        print(f"Warning: Ping error from servo: {packet_handler.getRxPacketError(error)}")
    else:
        print(f"Verified communication with servo {args.id} at {args.new_baud_rate} (model number: {model_number})")

    port_handler.closePort()
    print("Port closed")


if __name__ == "__main__":
    main()
