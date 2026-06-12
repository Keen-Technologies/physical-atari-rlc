cmake_minimum_required(VERSION 3.21)
project(robotroller)

set(CMAKE_CXX_STANDARD 17)

# Find Threads package (needed for std::thread on Linux)
find_package(Threads REQUIRED)

# Find OpenCV package (needed for camera capture)
find_package(OpenCV REQUIRED)

# Find pybind11 package (needed for Python bindings)
find_package(pybind11 REQUIRED)

# Find apriltag package (needed for AprilTag detection)
find_package(apriltag REQUIRED)

include_directories(/usr/local/include/dynamixel_sdk/)
link_directories(/usr/local/lib/)

# Build the Python module
pybind11_add_module(robotroller bindings.cpp src/servo.cpp src/robotroller.cpp src/camera.cpp src/apriltag_detector.cpp src/physical_atari_env.cpp)

# Link Python module against required libraries
target_link_libraries(robotroller PRIVATE Threads::Threads)
target_link_libraries(robotroller PRIVATE ${OpenCV_LIBS})
target_link_libraries(robotroller PRIVATE apriltag)

# Link Dynamixel SDK to Python module
if(APPLE)
    target_link_libraries(robotroller PUBLIC dxl_mac_cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_link_libraries(robotroller PUBLIC dxl_x64_cpp)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    target_link_libraries(robotroller PUBLIC dxl_sbc_cpp)
else()
    message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}, defaulting to dxl_x64_cpp")
    target_link_libraries(robotroller PUBLIC dxl_x64_cpp)
endif()

# Install the Python module to the active Python environment
execute_process(
    COMMAND python -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

install(TARGETS robotroller
        LIBRARY DESTINATION ${PYTHON_SITE_PACKAGES})

# Print installation instructions
message(STATUS "")
message(STATUS "To build and install the Python module:")
message(STATUS "  mkdir -p build && cd build")
message(STATUS "  cmake ..")
message(STATUS "  make")
message(STATUS "  sudo make install")
message(STATUS "")
