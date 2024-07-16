import os

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from ament_index_python.packages import get_package_share_directory

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    os.environ["SPOT_URDF_EXTRAS"] = os.path.join(get_package_share_directory('spot_driver_plus'), 'urdf', 'gen3.urdf.xacro')

    DISPLAY = os.getenv("DISPLAY")
    if DISPLAY == None:
        os.environ["DISPLAY"] = ''

    moveit_config = MoveItConfigsBuilder("spot", package_name="spot_gen3_moveit").planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner", "chomp"]
        ).moveit_cpp(
            file_path=get_package_share_directory("spot_gen3_moveit")
            + "/config/python_api.yaml"
        ).to_moveit_configs()

    follow_aruco_node = Node(
        package="spot_gen3_moveit",
        executable="follow_aruco.py",
        output="both",
        parameters=[moveit_config.to_dict()],
    )

    return LaunchDescription([follow_aruco_node])
