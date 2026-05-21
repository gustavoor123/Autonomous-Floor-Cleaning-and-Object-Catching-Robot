#!/bin/bash
sleep 2
ros2 param set /slam_toolbox minimum_travel_distance 0.1
ros2 param set /slam_toolbox minimum_travel_heading 0.1
ros2 param set /slam_toolbox map_update_interval 2.0
ros2 param set /slam_toolbox transform_timeout 1.0
ros2 param set /slam_toolbox tf_buffer_duration 60.0
echo "SLAM params updated"
