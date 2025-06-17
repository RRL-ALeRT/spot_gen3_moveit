import os

from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    os.environ["SPOT_URDF_EXTRAS"] = os.path.join(get_package_share_directory('spot_driver_plus'), 'urdf', 'gen3.urdf.xacro')

    DISPLAY = os.getenv("DISPLAY")
    if DISPLAY == None:
        os.environ["DISPLAY"] = ''

    moveit_config = MoveItConfigsBuilder("spot", package_name="spot_gen3_moveit").planning_pipelines(
            pipelines=["pilz_industrial_motion_planner"]).to_moveit_configs()
    return generate_move_group_launch(moveit_config)
