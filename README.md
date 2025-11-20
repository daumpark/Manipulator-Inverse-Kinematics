# [Manipulator Inverse Kinematics]

Manipulator Inverse Kinematics Lecture with Youtube Videos and Source Codes.  

## Overview
This repository provides beginner course of manipulator inverse kinematics.

## Dependencies

### 1. Prerequisites
* **ROS 2 Distribution**: Humble
* **Python**: 3.8+

### 2. Required Packages

* **Core & Build Tools**
    * `rclpy`: ROS 2 Python client library.
    * `ament_index_python`: Python API for the Ament resource index.
* **Math & Kinematics**
    * `pinocchio`: A fast and flexible implementation of Rigid Body Dynamics algorithms.
* **Messages & Interfaces**
    * `std_msgs` / `sensor_msgs`: Standard ROS 2 message definitions.
    * `visualization_msgs`: Messages for 3D visualization markers.
* **Visualization & Robot State**
    * `rviz2`: 3D visualization tool for ROS 2.
    * `interactive_markers`: Tools for creating interactive 3D markers in RViz.
    * `robot_state_publisher`: Publishes the state of the robot (tf tree) to `tf2`.

### 3. Installation

After cloning this repository into your workspace `src` folder, you can install all dependencies automatically using `rosdep`.

```bash
# Navigate to your workspace root (e.g., ~/ros2_ws)
cd ~/ros2_ws

# Update rosdep database
sudo apt update
rosdep update

# Install dependencies
rosdep install --from-paths src --ignore-src -r -y
sudo apt install ros-humble-pinocchio
```

## Usage
```bash
cd ~/ros2_ws
colcon build --symlink-install
source ~/.bashrc
```

1. **Analytical IK** – 2-DOF planar analytical IK via interactive markers in RViz  
   ```bash
   ros2 launch analytical_ik analytical_ik_test.launch.py
   ```
2. **Jacobian-based IK** – numerical IK via interactive markers in RViz  
   ```bash
   ros2 launch numerical_ik numerical_ik_test.launch.py solver:=<solver_name>
   ```
3. **Heuristic IK** – 2D heuristic IK with RViz visualization  
   ```bash
   ros2 launch heuristic_ik heuristic_ik_test.launch.py solver:=<solver_name>
   ```
4. **Redundancy & Null-space** – Real-time redundant IK using CLIK-style solvers in RViz 
   ```bash
   ros2 launch redundant_ik redundant_ik_test.launch.py solver:=<solver_name>
   ```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact
Maintainer: [Daum Park] (doumpork@khu.ac.kr)  
Lab: [RCI Lab @ Kyung Hee University](https://rcilab.khu.ac.kr)
