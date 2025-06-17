import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from geometry_msgs.msg import TransformStamped
import tf2_ros
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
from tf2_ros import TransformBroadcaster
import tf_transformations

PIPE_DETECTED = False  


class DetectPipe(Node):
    def __init__(self):
        super().__init__("detect_pipe")
        self.color_subscription = self.create_subscription(
            Image, "/kinova_color", self.image_callback, 10
        )
        self.depth_subscription = self.create_subscription(
            Image, "/depth_registered", self.depth_callback, 10
        )

        self.bridge = CvBridge()
        self.latest_color_frame = None
        self.latest_depth_frame = None
        self.color_circles = None
        self.depth_circles = None

        self.color_hough_dp = 1.0              # Slightly higher for better accuracy
        self.color_hough_min_dist = 40         # Increase to avoid multiple detections of same circle
        self.color_hough_param1 = 100          # Higher Canny threshold for less noise
        self.color_hough_param2 = 40          # Lower accumulator threshold for more sensitivity
        self.color_min_radius = 12             # Adjust based on expected pipe size in pixels
        self.color_max_radius = 50           # Adjust based on expected pipe size in pixels

        # Depth Hough parameters (may need tuning)
        self.depth_hough_dp = 1.0
        self.depth_hough_min_dist = 30
        self.depth_hough_param1 = 100
        self.depth_hough_param2 = 40
        self.depth_min_radius = 15
        self.depth_max_radius = 50

        # Scaling factors (based on actual resolutions)
        self.depth_width = 640
        self.depth_height = 480
        self.color_width = 640
        self.color_height = 480
        self.scale_x = self.depth_width / self.color_width
        self.scale_y = self.depth_height / self.color_height

        # Dynamic offsets
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.max_offset_distance = 100.0  # Max pixel distance for matching circles

        # TF setup
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.base_frame = "map"  
        self.camera_frame = "camera_link"

        self.get_logger().info("DetectPipe node has been started.")
        

    def find_matching_circle(self, x_color, y_color, r_color):
        """Find the closest depth circle to a color circle."""
        if self.depth_circles is None:
            return None, 0.0, 0.0

        min_distance = float("inf")
        best_match = None
        offset_x, offset_y = 0.0, 0.0

        for x_depth, y_depth, r_depth in self.depth_circles:
            distance = np.sqrt((x_depth - x_color) ** 2 + (y_depth - y_color) ** 2)
            radius_diff = abs(r_depth - r_color)

            # Match based on proximity and similar radius
            if (
                distance < min_distance
                and distance < self.max_offset_distance
                and radius_diff < 10
            ):
                min_distance = distance
                best_match = (x_depth, y_depth, r_depth)
                offset_x = x_depth - x_color
                offset_y = y_depth - y_color

        return best_match, offset_x, offset_y

    def image_callback(self, msg):
        self.latest_color_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = self.latest_color_frame.copy()

        self.get_logger().debug(f"Color image shape: {frame.shape}")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (9, 9), 2)

        self.color_circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=self.color_hough_dp,
            minDist=self.color_hough_min_dist,
            param1=self.color_hough_param1,
            param2=self.color_hough_param2,
            minRadius=self.color_min_radius,
            maxRadius=self.color_max_radius,
        )

        if self.color_circles is not None:
            self.color_circles = np.round(self.color_circles[0, :]).astype("int")
            for x, y, r in self.color_circles:
                # Draw on color image (use color image coordinates)
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.rectangle(
                    frame,
                    (x - 5, y - 5),
                    (x + 5, y + 5),
                    (0, 128, 255),
                    -1,
                )

                # If you want to annotate on the depth image, scale coordinates:
                x_scaled = int(x * self.scale_x)
                y_scaled = int(y * self.scale_y)
                r_scaled = int(r * self.scale_x)
                # Optionally, only add offsets if a matching depth circle is found
                # _, offset_x, offset_y = self.find_matching_circle(x_scaled, y_scaled, r_scaled)
                # x_depth = int(x_scaled + offset_x)
                # y_depth = int(y_scaled + offset_y)
                # Otherwise, just use x_scaled, y_scaled

                # You can store these for use in depth_callback if needed

                # Check depth
                if self.latest_depth_frame is not None:
                    if (
                        0 <= int(y_scaled) < self.depth_height
                        and 0 <= int(x_scaled) < self.depth_width
                    ):
                        y_min = max(0, int(y_scaled - r_scaled))
                        y_max = min(self.depth_height, int(y_scaled + r_scaled))
                        x_min = max(0, int(x_scaled - r_scaled))
                        x_max = min(self.depth_width, int(x_scaled + r_scaled))
                        depth_roi = self.latest_depth_frame[y_min:y_max, x_min:x_max]

                        if depth_roi.size == 0:
                            self.get_logger().warn(
                                f"Empty depth ROI at ({x_scaled}, {y_scaled})"
                            )
                            continue

                        valid_depths = depth_roi[depth_roi > 0]
                        if valid_depths.size == 0:
                            self.get_logger().warn(
                                f"No valid depth values in ROI at ({x_scaled}, {y_scaled})"
                            )
                            continue

                        avg_depth = np.mean(valid_depths)
                        distance_meters = avg_depth / 1000.0  # Convert mm to meters

                        if distance_meters < 1.0:
                            self.get_logger().debug(
                                f"Circle at ({x}, {y}) [depth coords: ({x_scaled}, {y_scaled})], "
                                f"radius {r} [scaled: {r_scaled}], "
                                f"avg distance: {distance_meters:.2f} meters"
                            )

                        else:
                            self.get_logger().debug(
                                f"Circle at ({x}, {y}) with avg distance {distance_meters:.2f} meters ignored (>= 1 meter)"
                            )
                    else:
                        self.get_logger().warn(
                            f"Scaled circle at ({x_scaled}, {y_scaled}) out of depth bounds ({self.depth_width}x{self.depth_height})"
                        )
                else:
                    self.get_logger().debug(
                        f"Circle at ({x}, {y}) with radius {r} (no depth yet)"
                    )

        else:
            self.get_logger().debug("No circles detected in color image")

        if self.latest_depth_frame is not None:
            depth_display = cv2.normalize(
                self.latest_depth_frame, None, 0, 255, cv2.NORM_MINMAX
            )
            depth_display = cv2.convertScaleAbs(depth_display)
            depth_display = cv2.cvtColor(depth_display, cv2.COLOR_GRAY2BGR)
            depth_display_resized = cv2.resize(depth_display, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            blended = cv2.addWeighted(frame, 0.5, depth_display_resized, 0.5, 0.0)
        #     cv2.imshow("Blended Color and Depth", blended)
        cv2.imshow("Color - Detected Circles", frame)
        cv2.waitKey(1)

    def depth_callback(self, msg):
        global PIPE_DETECTED
        self.latest_depth_frame = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        if self.latest_depth_frame is None or self.latest_depth_frame.size == 0:
            self.get_logger().error("Error: depth image is empty")
            return

        self.get_logger().debug(f"Depth image shape: {self.latest_depth_frame.shape}")

        # depth image for circle detection
        depth_norm = cv2.normalize(
            self.latest_depth_frame, None, 0, 255, cv2.NORM_MINMAX
        )
        depth_gray = cv2.convertScaleAbs(depth_norm)
        depth_gray = cv2.GaussianBlur(depth_gray, (5, 5), 0)

        self.depth_circles = cv2.HoughCircles(
            depth_gray,
            cv2.HOUGH_GRADIENT,
            dp=self.depth_hough_dp,
            minDist=self.depth_hough_min_dist,
            param1=self.depth_hough_param1,
            param2=self.depth_hough_param2,
            minRadius=self.depth_min_radius,
            maxRadius=self.depth_max_radius,
        )

        depth_display = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)

        if self.depth_circles is not None:
            self.depth_circles = np.round(self.depth_circles[0, :]).astype("int")
            for x, y, r in self.depth_circles:
                cv2.circle(depth_display, (x, y), r, (255, 0, 0), 4)
                cv2.rectangle(
                    depth_display,
                    (x - 5, y - 5),
                    (x + 5, y + 5),
                    (255, 128, 0),
                    -1,
                )
        else:
            self.get_logger().debug("No circles detected in depth image")

        if self.color_circles is not None:
            height, width = self.latest_depth_frame.shape[:2]
            for x, y, r in self.color_circles:
                x_scaled = float(x * self.scale_x)
                y_scaled = float(y * self.scale_y)
                r_scaled = float(r * self.scale_x)

                x_depth = x_scaled + self.offset_x
                y_depth = y_scaled + self.offset_y

                x_depth_int = int(x_depth)
                y_depth_int = int(y_depth)
                r_scaled_int = int(r_scaled)

                if 0 <= x_depth_int < width and 0 < y_depth_int < height:
                    PIPE_DETECTED = True
                    depth_value = self.latest_depth_frame[y_depth_int, x_depth_int]
                    distance_meters = float(depth_value) / 1000.0  # Convert mm to meters

                    if 0 < distance_meters <= 1.0:
                        
                        cv2.circle(
                            depth_display,
                            (x_depth_int, y_depth_int),
                            r_scaled_int,
                            (0, 255, 0),
                            4,
                        )
                        cv2.rectangle(
                            depth_display,
                            (x_depth_int - 5, y_depth_int - 5),
                            (x_depth_int + 5, y_depth_int + 5),
                            (0, 128, 255),
                            -1,
                        )
                        self.broadcast_pipe_transform(
                            x_depth, y_depth, depth_value
                        )
                        PIPE_DETECTED = False
                    else:
                        self.get_logger().debug(
                            f"Skipped circle at ({x_depth_int}, {y_depth_int}) with distance {distance_meters:.2f} m"
                        )
                else:
                    self.get_logger().debug(f"Skipped out of boumd circle at ({x_depth_int}, {y_depth_int})")

        # cv2.imshow("Depth - Overlay", depth_display)
        # cv2.waitKey(1)

    def broadcast_pipe_transform(self, cX, cY, depth):
        fx = 653.68
        fy = 651.86
        cx = 311.75
        cy = 232.4

        depth = float(depth) / 1000.0   # Ensure depth is a float and in meters

        X = (cX - cx) * depth / fx
        Y = (cY - cy) * depth / fy
        Z = depth

        self.get_logger().debug(
            f"pipe in {self.camera_frame}: x={X:.3f}, y={Y:.3f}, z={Z:.3f}"
        )

        cam_pos = np.array([X, Y, Z], dtype=float)

        try:
            if not self.tf_buffer.can_transform(
                self.base_frame,
                self.camera_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0),
            ):
                self.get_logger().error(
                    f"Transform from {self.camera_frame} to {self.base_frame} not available"
                )
                return

            transform = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.05),
            )
            self.get_logger().debug(
                f"Transform: translation=({transform.transform.translation.x:.3f}, "
                f"{transform.transform.translation.y:.3f}, {transform.transform.translation.z:.3f}), "
                f"rotation=({transform.transform.rotation.x:.3f}, {transform.transform.rotation.y:.3f}, "
                f"{transform.transform.rotation.z:.3f}, {transform.transform.rotation.w:.3f})"
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ],
                dtype=float,
            )

            quaternion = [
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w,
            ]

            rotation_matrix = tf_transformations.quaternion_matrix(quaternion)[:3, :3]

            base_pos = np.dot(rotation_matrix, cam_pos) + translation

            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = self.base_frame
            tf_msg.child_frame_id = f"pipe"
            tf_msg.transform.translation.x = base_pos[0]
            tf_msg.transform.translation.y = base_pos[1]
            tf_msg.transform.translation.z = base_pos[2]
            tf_msg.transform.rotation.x = 0.0
            tf_msg.transform.rotation.y = 0.0
            tf_msg.transform.rotation.z = 0.0
            tf_msg.transform.rotation.w = 1.0
        
            self.tf_broadcaster.sendTransform(tf_msg)
            self.get_logger().info(
                f"[TF] pipe in {self.base_frame}: x={base_pos[0]:.3f}, "
                f"y={base_pos[1]:.3f}, z={base_pos[2]:.3f}"
            )

        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().error(f"TF lookup failed: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in TF broadcast: {e}")


def main(args=None):
    try:
        rclpy.init(args=args)
        detect_pipe = DetectPipe()
        rclpy.spin(detect_pipe)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            detect_pipe.destroy_node()
            rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
