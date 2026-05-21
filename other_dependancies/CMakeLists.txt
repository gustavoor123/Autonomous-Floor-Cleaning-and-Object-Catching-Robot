cmake_minimum_required(VERSION 3.5)
project(my_bot)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY config description launch maps rviz worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/publish_robot_description.py
    scripts/arduino_bridge.py
    scripts/robot_mode_manager.py
    scripts/joy_mode_switch.py
    scripts/trash_detector.py
    scripts/trash_approach.py
    scripts/cmd_vel_mux.py
    scripts/nav_wander.py
    scripts/odom_tf_relay.py
    scripts/scan_relay.py
    scripts/scan_restamper.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
