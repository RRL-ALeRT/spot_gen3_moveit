#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from cv_bridge import CvBridge
import cv2
import numpy as np
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Twist, TwistStamped, Quaternion
import tf2_ros
import tf2_py
import transforms3d as t3d
from transforms3d.euler import quat2euler
from geometry_msgs.msg import Vector3
from tf2_geometry_msgs import do_transform_pose
import tf2_geometry_msgs
from tf2_ros import StaticTransformBroadcaster

from moveit_msgs.msg import ServoStatus

import time
import math
from scipy.spatial.transform import Rotation as R

from world_info_msgs.msg import BoundingPolygon, BoundingPolygonArray
from geometry_msgs.msg import Point32

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState
from moveit_msgs.srv import GetPositionIK

from geometry_msgs.msg import Pose, Quaternion, TransformStamped, Twist

import numpy as np
import time

from tf2_geometry_msgs.tf2_geometry_msgs import do_transform_pose

from rclpy.action import ActionClient
from rclpy.node import Node

from moveit_msgs.msg import MotionPlanRequest
from moveit_msgs.msg import JointConstraint
from moveit_msgs.msg import Constraints
from moveit_msgs.msg import PlanningOptions
from moveit_msgs.action import MoveGroup

from copy import deepcopy

ARUCO_ID = 21
ARUCO_SIZE = 0.04
DISTANCE_TO_ARUCO = 0.28


class Moveit(Node):
    def __init__(self):
        super().__init__("moveit_plan_execute_python")

        self._action_client = ActionClient(self, MoveGroup, "/move_action")

    def send_goal(self, target_angles):
        self.joint_state = None
        self.goal_done = False

        motion_plan_request = MotionPlanRequest()

        motion_plan_request.workspace_parameters.header.stamp = (
            self.get_clock().now().to_msg()
        )
        motion_plan_request.workspace_parameters.header.frame_id = "gen3_base_link"
        motion_plan_request.workspace_parameters.min_corner.x = -1.0
        motion_plan_request.workspace_parameters.min_corner.y = -1.0
        motion_plan_request.workspace_parameters.min_corner.z = -1.0
        motion_plan_request.workspace_parameters.max_corner.x = 1.0
        motion_plan_request.workspace_parameters.max_corner.y = 1.0
        motion_plan_request.workspace_parameters.max_corner.z = 1.0
        motion_plan_request.start_state.is_diff = True

        jc = JointConstraint()
        jc.tolerance_above = 0.01
        jc.tolerance_below = 0.01
        jc.weight = 1.0

        joints = {}
        joints["joint_1"] = target_angles[0]
        joints["joint_2"] = target_angles[1]
        joints["joint_3"] = target_angles[2]
        joints["joint_4"] = target_angles[3]
        joints["joint_5"] = target_angles[4]
        joints["joint_6"] = target_angles[5]

        constraints = Constraints()
        for joint, angle in joints.items():
            jc.joint_name = joint
            jc.position = angle
            constraints.joint_constraints.append(deepcopy(jc))

        motion_plan_request.goal_constraints.append(constraints)

        motion_plan_request.pipeline_id = "ompl"
        # motion_plan_request.planner_id = "STOMP"
        motion_plan_request.group_name = "manipulator"
        motion_plan_request.num_planning_attempts = 10
        motion_plan_request.allowed_planning_time = 10.0
        motion_plan_request.max_velocity_scaling_factor = 0.5
        motion_plan_request.max_acceleration_scaling_factor = 0.5
        motion_plan_request.max_cartesian_speed = 0.0

        planning_options = PlanningOptions()
        planning_options.plan_only = False
        planning_options.look_around = False
        planning_options.look_around_attempts = 0
        planning_options.max_safe_execution_cost = 0.0
        planning_options.replan = True
        planning_options.replan_attempts = 10
        planning_options.replan_delay = 0.1

        goal_msg = MoveGroup.Goal()
        goal_msg.request = motion_plan_request
        goal_msg.planning_options = planning_options

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback
        )
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected :(")
            self.goal_done = True
            return

        self.get_logger().info("Goal accepted :)")

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        # self.get_logger().info(str(future))
        self.goal_done = True

    def feedback_callback(self, feedback_msg):
        # self.get_logger().info(str(feedback_msg))
        pass


relative_points = np.array(
    [
        [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
        [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
        [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
    ],
    dtype=np.float32,
)


class IK(Node):
    def __init__(self):
        super().__init__("moveit_ik")

        self.create_subscription(JointState, "/joint_states", self.joint_states_cb, 1)

        self.cli = self.create_client(GetPositionIK, "/compute_ik")

        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
            rclpy.spin_once(self)

    def joint_states_cb(self, joint_state):
        self.joint_state = joint_state

    def send_request(self, pose):
        self.joint_state = None

        while self.joint_state is None:
            rclpy.spin_once(self)

        req = GetPositionIK.Request()
        
        req.ik_request.group_name = "manipulator"
        req.ik_request.robot_state.joint_state = self.joint_state
        req.ik_request.avoid_collisions = True

        req.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        req.ik_request.pose_stamped.header.frame_id = "gen3_base_link"

        req.ik_request.pose_stamped.pose = pose
        req.ik_request.timeout.sec = 10

        self.future = self.cli.call_async(req)
        rclpy.spin_until_future_complete(self, self.future)

        target_angles = list(self.future.result().solution.joint_state.position)[:6]

        if len(target_angles) > 0:
            return target_angles

        self.get_logger().warn("No ik soln found")
        return None


def moveit_motion(pose):
    ik = IK()
    moveit = Moveit()
    target_angles = ik.send_request(pose)
    ik.destroy_node()
    if target_angles != None:
        moveit.send_goal(target_angles)
        while not moveit.goal_done:
            rclpy.spin_once(moveit)
        moveit.destroy_node()
        return target_angles
    moveit.destroy_node()
    return target_angles


def compute_end_effector_twist(body_twist, body_to_ee_transform, ee_to_body_transform):
    # Convert Quaternion to Rotation Matrix for both transformations
    quat_ab = [body_to_ee_transform.transform.rotation.x,
               body_to_ee_transform.transform.rotation.y,
               body_to_ee_transform.transform.rotation.z,
               body_to_ee_transform.transform.rotation.w]
    r_ab = R.from_quat(quat_ab)
    rotation_matrix_ab = r_ab.as_matrix()

    quat_ba = [ee_to_body_transform.transform.rotation.x,
               ee_to_body_transform.transform.rotation.y,
               ee_to_body_transform.transform.rotation.z,
               ee_to_body_transform.transform.rotation.w]
    r_ba = R.from_quat(quat_ba)
    rotation_matrix_ba = r_ba.as_matrix()

    # Calculate translation for both transformations
    translation_ab = np.array([body_to_ee_transform.transform.translation.x,
                                body_to_ee_transform.transform.translation.y,
                                body_to_ee_transform.transform.translation.z])


class CameraLinkPose:
    def __init__(self, tf_buffer, logger):
        self.logger = logger
        self.tf_buffer = tf_buffer

        self.nominal_x_position = 0.617
        self.min_x_position = 0.61
        self.max_x_position = 0.8

        self.nominal_y_position = 0.0
        self.min_y_position = self.nominal_y_position - 0.15
        self.max_y_position = self.nominal_y_position + 0.15

        self.nominal_z_position = 0.6
        self.min_z_position = self.nominal_z_position - 0.1
        self.max_z_position = self.nominal_z_position + 0.1

        self.source = "camera_link"
        self.target = "body"

        self.height_source = "camera_link"
        self.height_target = "gpe"

        self.current_spot_height = 0.0
        self.new_spot_height = 0.0

    def is_ee_within_xyz_limit(self):
        SPOT_SPEED_TR = 0.16
        SPOT_SPEED_ROT = 0.16

        change_height = False

        try:
            t = self.tf_buffer.lookup_transform(
                self.target, self.source, rclpy.time.Time()
            )

            x = 0.0
            if t.transform.translation.x < self.min_x_position:
                x = -SPOT_SPEED_TR
            elif t.transform.translation.x > self.max_x_position:
                x = SPOT_SPEED_TR

            y = 0.0
            if t.transform.translation.y < self.min_y_position:
                y = -SPOT_SPEED_ROT
            elif t.transform.translation.y > self.max_y_position:
                y = SPOT_SPEED_ROT

            DELTA_HEIGHT = 0.005
            if t.transform.translation.z < self.min_z_position:
                self.new_spot_height -= DELTA_HEIGHT
            elif t.transform.translation.z > self.max_z_position:
                self.new_spot_height += DELTA_HEIGHT
            
            # self.logger.info(f"{self.new_spot_height}")

            MIN_SPOT_HEIGHT = -0.05
            MAX_SPOT_HEIGHT = 0.05
            self.new_spot_height = np.clip(self.new_spot_height, MIN_SPOT_HEIGHT, MAX_SPOT_HEIGHT)

            t1 = self.tf_buffer.lookup_transform(
                self.source, self.target, rclpy.time.Time()
            )

            if x != 0 or y != 0 or self.current_spot_height != self.new_spot_height:
                if self.current_spot_height != self.new_spot_height:
                    self.current_spot_height = self.new_spot_height
                    change_height = True
                return False, change_height, x, y, self.current_spot_height, [t1, t]

            return True, change_height, 0.0, 0.0, self.current_spot_height, [t1, t]

        except tf2_ros.TransformException as ex:
            # self.get_logger().info(f"Could not transform {self.source} to {self.target}: {ex}")
            pass

        return False, change_height, 0.0, 0.0, self.current_spot_height, None


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector_node')

        static_broadcaster = StaticTransformBroadcaster(self)

        transform1 = TransformStamped()
        transform1.header.frame_id = 'aruco_21'
        transform1.child_frame_id = 'camera_link_target'
        transform1.transform.translation.z = DISTANCE_TO_ARUCO
        transform1.transform.rotation.y = 1.0
        transform1.transform.rotation.w = 0.0

        transform2 = TransformStamped()
        transform2.header.frame_id = 'camera_link_target'
        transform2.child_frame_id = 'tool_frame_target'
        transform2.transform.translation.y = 0.056
        transform2.transform.translation.z = 0.203
        transform2.transform.rotation.z = 1.0
        transform2.transform.rotation.w = 0.0

        static_broadcaster.sendTransform([transform1, transform2])

        self.bridge = CvBridge()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image, '/kinova_color', self.image_callback, 1)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, '/kinova_color/camera_info', self.camera_info_callback, 1)

        self.servo_status_sub = self.create_subscription(ServoStatus, '/servo_node/status', self.servo_status_cb, 1)

        # self.create_timer(1/50, self.step)
        self.ee_twist_pub = self.create_publisher(TwistStamped, "/twist_controller/commands", 1)
        self.spot_twist_pub = self.create_publisher(Twist, "/cmd_vel", 1)
        self.spot_body_pose_pub = self.create_publisher(Pose, "/body_pose", 1)

        self.bounding_polygon_pub = self.create_publisher(BoundingPolygonArray, "/kinova_color/bp", 1)

        ar_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(ar_dict)

        self.source = 'tool_frame_target'
        self.target = 'gen3_base_link'

        # Initialize previous values for smoothing
        self.prev_linear = np.zeros(3)
        self.prev_angular = np.zeros(3)
        self.alpha = 0.2  # Smoothing factor (tune this value)

        self.target_goal_reached = False

        self.camera_pose = CameraLinkPose(self.tf_buffer, self.get_logger())

        self.pause = False

        # self.logger = get_logger("moveit_py.pose_goal")

        # instantiate MoveItPy instance and get planning component
        # self.panda = MoveItPy(node_name="moveit_py")
        # self.panda_arm = self.panda.get_planning_component("manipulator")
        # self.logger.info("MoveItPy instance created")

    def servo_status_cb(self, msg):
        status = msg.code
        if status == 0:
            self.pause = False
            return

        self.pause = True
        self.get_logger().warn(f"{status}")
        time.sleep(0.1)

    def image_callback(self, msg):
        if not hasattr(self, 'dist_coeff'):
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        (corners, ids, rejected) = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            for i in range(len(ids)):
                if ids[i] == ARUCO_ID:
                    bounding_polygon_msg = BoundingPolygonArray()
                    bounding_polygon_msg.header = msg.header
                    bounding_polygon_msg.type = "aruco"

                    bounding_polygon = BoundingPolygon()
                    bounding_polygon.name = "aruco_21"

                    for corner in corners[i][0]:
                        point = Point32()
                        point.x = float(corner[0])
                        point.y = float(corner[1])
                        bounding_polygon.array.append(point)

                    bounding_polygon_msg.array.append(bounding_polygon)

                    self.bounding_polygon_pub.publish(bounding_polygon_msg)

                    _, rvec, tvec = cv2.solvePnP(
                        relative_points, corners[i], self.camera_matrix, self.dist_coeff
                    )
                    transform = self.calculate_transform(rvec, tvec)
                    self.publish_transform(transform, msg.header, ARUCO_ID)

        self.step()

    def camera_info_callback(self, msg):
        if hasattr(self, 'dist_coeff'):
            return
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeff = np.array(msg.d)

    def calculate_transform(self, rvec, tvec):
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_vector = np.array(tvec).reshape((3, 1))

        transform = np.eye(4)
        transform[:3, :3] = rotation_matrix
        transform[:3, 3] = translation_vector.flatten()

        return transform

    def publish_transform(self, transform, header, aruco_id):
        tf_msg = TransformStamped()
        tf_msg.header = header
        tf_msg.child_frame_id = f'aruco_{aruco_id}'
        tf_msg.transform.translation.x = transform[0, 3]
        tf_msg.transform.translation.y = transform[1, 3]
        tf_msg.transform.translation.z = transform[2, 3]

        rotation_matrix = transform[:3, :3]
        rotation_quaternion = t3d.quaternions.mat2quat(rotation_matrix)
        tf_msg.transform.rotation.x = -rotation_quaternion[2]
        tf_msg.transform.rotation.y = rotation_quaternion[1]
        tf_msg.transform.rotation.z = -rotation_quaternion[0]
        tf_msg.transform.rotation.w = rotation_quaternion[3]

        self.tf_broadcaster.sendTransform(tf_msg)

    def step(self):
        GOAL_DISTANCE_TOLERANCE = 0.05
        GOAL_ANGULAR_TOLERANCE = 0.12

        ee_twist_msg = None

        ee_within_xyz_limit, change_height, spot_speed_x, spot_yaw, spot_height, t_body_camera_link = self.camera_pose.is_ee_within_xyz_limit()
        # self.get_logger().info(f"{ee_within_xyz_limit}, {x}")
        ee_within_xyz_limit = True
        if not ee_within_xyz_limit:
            if spot_speed_x != 0 or spot_yaw != 0 or change_height:
                spot_twist_msg = Twist()
                spot_twist_msg.linear.x = spot_speed_x
                spot_twist_msg.angular.z = spot_yaw
                # spot_twist_msg.linear.y = spot_yaw

                body_pose_msg = Pose()
                # self.get_logger().info(f"{spot_height}")
                body_pose_msg.position.z = spot_height

                if not self.pause:
                    # self.spot_body_pose_pub.publish(body_pose_msg)
                    self.spot_twist_pub.publish(spot_twist_msg)

                    ee_twist_msg = TwistStamped()
                    ee_twist_msg.header.frame_id = "body"
                    ee_twist_msg.header.stamp = self.get_clock().now().to_msg()

                    ee_twist_msg.twist.linear.x = -1.2*spot_twist_msg.linear.x
                    ee_twist_msg.twist.linear.y = -1.2*spot_twist_msg.angular.z

                    if t_body_camera_link is not None:
                        t_body_camera_link, t_camera_link_body = t_body_camera_link
                        # ee_twist_msg.twist = compute_end_effector_twist(spot_twist_msg, t_body_camera_link, t_camera_link_body)

                        # print(spot_twist_msg, ee_twist_msg)

        try:
            t = self.tf_buffer.lookup_transform(
                self.target, self.source, rclpy.time.Time()
            )

            within_distance_tolerance = False
            if abs(t.transform.translation.x) < GOAL_DISTANCE_TOLERANCE and \
            abs(t.transform.translation.y) < GOAL_DISTANCE_TOLERANCE and \
            abs(t.transform.translation.z) < GOAL_DISTANCE_TOLERANCE:
                within_distance_tolerance = True

            # within_angular_tolerance = False
            # if abs(euler_angles[0]) < GOAL_ANGULAR_TOLERANCE and \
            # abs(euler_angles[1]) < GOAL_ANGULAR_TOLERANCE and \
            # abs(euler_angles[2]) < GOAL_ANGULAR_TOLERANCE:
            #     within_angular_tolerance = True

            # if within_distance_tolerance:# and within_angular_tolerance:
            #     self.tf_buffer.clear()
            #     return

            # set plan start state to current state
            # self.panda_arm.set_start_state_to_current_state()

            if not within_distance_tolerance:
                pose_goal = Pose()
                pose_goal.position.x = t.transform.translation.x
                pose_goal.position.y = t.transform.translation.y 
                pose_goal.position.z = t.transform.translation.z 
                pose_goal.orientation.x = t.transform.rotation.x
                pose_goal.orientation.y = t.transform.rotation.y
                pose_goal.orientation.z = t.transform.rotation.z
                pose_goal.orientation.w = t.transform.rotation.w

                moveit_motion(pose_goal)
            self.tf_buffer.clear()

            # self.panda_arm.set_goal_state(pose_stamped_msg=pose_goal, pose_link="camera_link")

            # plan to goal
            # plan_and_execute(self.panda, self.panda_arm, self.logger, sleep_time=0.1)

        #     # Apply low-pass filter to angular components
        #     euler_angles = quat2euler([
        #         t.transform.rotation.w,
        #         t.transform.rotation.x,
        #         t.transform.rotation.y,
        #         t.transform.rotation.z
        #     ])

        #     within_angular_tolerance = False
        #     if abs(euler_angles[0]) < GOAL_ANGULAR_TOLERANCE and \
        #     abs(euler_angles[1]) < GOAL_ANGULAR_TOLERANCE and \
        #     abs(euler_angles[2]) < GOAL_ANGULAR_TOLERANCE:
        #         within_angular_tolerance = True
                
        #     if ee_twist_msg is None:
        #         ee_twist_msg = TwistStamped()
        #         ee_twist_msg.header = t.header

        #         if not within_distance_tolerance:
        #             # Apply low-pass filter to linear components
        #             ee_twist_msg.twist.linear.x = 1.5 * self.alpha * t.transform.translation.x + (1 - self.alpha) * self.prev_linear[0]
        #             ee_twist_msg.twist.linear.y = 1.5 * self.alpha * t.transform.translation.y + (1 - self.alpha) * self.prev_linear[1]
        #             ee_twist_msg.twist.linear.z = 1.5 * self.alpha * t.transform.translation.z + (1 - self.alpha) * self.prev_linear[2]
                    
        #             linear_max = 0.2
        #             ee_twist_msg.twist.linear.x = np.clip(ee_twist_msg.twist.linear.x, -linear_max, linear_max)
        #             ee_twist_msg.twist.linear.y = np.clip(ee_twist_msg.twist.linear.y, -linear_max, linear_max)
        #             ee_twist_msg.twist.linear.z = np.clip(ee_twist_msg.twist.linear.z, -linear_max, linear_max)

        #         if not within_angular_tolerance:
        #             ee_twist_msg.twist.angular.x = 1.5 * self.alpha * euler_angles[0] + (1 - self.alpha) * self.prev_angular[0]
        #             ee_twist_msg.twist.angular.y = 1.5 * self.alpha * euler_angles[1] + (1 - self.alpha) * self.prev_angular[1]
        #             ee_twist_msg.twist.angular.z = 1.5 * self.alpha * euler_angles[2] + (1 - self.alpha) * self.prev_angular[2]

        #             angular_max = 0.8

        #             ee_twist_msg.twist.angular.x = np.clip(ee_twist_msg.twist.angular.x, -angular_max, angular_max)
        #             ee_twist_msg.twist.angular.y = np.clip(ee_twist_msg.twist.angular.y, -angular_max, angular_max)
        #             ee_twist_msg.twist.angular.z = np.clip(ee_twist_msg.twist.angular.z, -angular_max, angular_max)

        #     # self.get_logger().info(f"{within_distance_tolerance}, {within_angular_tolerance}")
        #     if within_distance_tolerance and within_angular_tolerance:
        #         self.target_goal_reached = True
        #     else:
        #         self.target_goal_reached = False

        except tf2_ros.TransformException as ex:
            pass
            self.get_logger().info(f"Could not transform {self.source} to {self.target}: {ex}")

        #     if ee_twist_msg is None:
        #         # Initialize ee_twist_msg with previous decayed values when transform is not available
        #         ee_twist_msg = TwistStamped()
        #         ee_twist_msg.header.frame_id = "camera_link"
        #         ee_twist_msg.header.stamp = self.get_clock().now().to_msg()
        #         ee_twist_msg.twist.linear.x = self.prev_linear[0] * 0.5  # Adjust the decay rate as needed
        #         ee_twist_msg.twist.linear.y = self.prev_linear[1] * 0.5  # Adjust the decay rate as needed
        #         ee_twist_msg.twist.linear.z = self.prev_linear[2] * 0.5  # Adjust the decay rate as needed
        #         ee_twist_msg.twist.angular.x = self.prev_angular[0] * 0.5  # Adjust the decay rate as needed
        #         ee_twist_msg.twist.angular.y = self.prev_angular[1] * 0.5  # Adjust the decay rate as needed
        #         ee_twist_msg.twist.angular.z = self.prev_angular[2] * 0.5  # Adjust the decay rate as needed

        # # Publish the result
        # if not self.target_goal_reached:
        #     # Update previous values for the next iteration
        #     self.prev_linear = np.array([ee_twist_msg.twist.linear.x, ee_twist_msg.twist.linear.y, ee_twist_msg.twist.linear.z])
        #     self.prev_angular = np.array([ee_twist_msg.twist.angular.x, ee_twist_msg.twist.angular.y, ee_twist_msg.twist.angular.z])

        # self.ee_twist_pub.publish(ee_twist_msg)
        # self.tf_buffer.clear()


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()

    # executor = MultiThreadedExecutor()
    # executor.add_node(node)
    # executor.spin()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == '__main__':
    main()
