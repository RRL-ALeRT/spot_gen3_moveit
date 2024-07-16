import os
import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_param_builder import ParameterBuilder
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    os.environ["SPOT_URDF_EXTRAS"] = os.path.join(get_package_share_directory('spot_driver_plus'), 'urdf', 'gen3.urdf.xacro')

    moveit_config = MoveItConfigsBuilder("spot", package_name="spot_gen3_moveit").to_moveit_configs()

    # Get parameters for the Servo node
    servo_params = {
        "moveit_servo": ParameterBuilder("kortex_bringup")
        .yaml("config/servo.yaml")
        .to_dict()
    }

    # This filter parameter should be >1. Increase it for greater smoothing but slower motion.
    low_pass_filter_coeff = {"butterworth_filter_coeff": 1.5}

    # This sets the update rate and planning group name for the acceleration limiting filter.
    acceleration_filter_update_period = {"update_period": 0.01}

    planning_group_name = {"planning_group_name": "manipulator"}

    servo_node = launch_ros.actions.Node(
        package="moveit_servo",
        executable="servo_node",
        name="servo_node",
        parameters=[
            servo_params,
            acceleration_filter_update_period,
            planning_group_name,
            low_pass_filter_coeff,
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
        ],
        output="screen",
    )

    # servo_node = launch_ros.actions.Node(
    #     package="moveit_servo",
    #     executable="demo_pose",
    #     name="demo_pose",
    #     parameters=[
    #         servo_params,
    #         acceleration_filter_update_period,
    #         planning_group_name,
    #         low_pass_filter_coeff,
    #         moveit_config.robot_description,
    #         moveit_config.robot_description_semantic,
    #         moveit_config.robot_description_kinematics,
    #         moveit_config.joint_limits,
    #     ],
    #     output="screen",
    # )

    return launch.LaunchDescription([servo_node])
