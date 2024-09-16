import rclpy
from rclpy.node import Node
from nav2_msgs.action import NavigateToPose
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.task import Future
from cv_bridge import CvBridge
import cv2
import os
import math
import tf_transformations
import csv

import random

class MapChecker(Node):
    def __init__(self):
        super().__init__('map_checker')

        # Subscribe to the /map topic to get the occupancy grid
        self.map_subscription = self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            10)

        # Store the map data
        self.map_data = None
        self.map_resolution = None
        self.map_origin_x = None
        self.map_origin_y = None
        self.map_width = None
        self.map_height = None

    def map_callback(self, msg):
        # Store the map data and relevant information
        self.map_data = msg.data
        self.map_resolution = msg.info.resolution  # Map resolution in meters/cell
        self.map_origin_x = msg.info.origin.position.x  # Origin of the map (bottom-left corner)
        self.map_origin_y = msg.info.origin.position.y
        self.map_width = msg.info.width  # Map width in cells
        self.map_height = msg.info.height  # Map height in cells

        self.get_logger().info('Map received and stored.')

    def is_position_free(self, x, y):
        if self.map_data is None:
            self.get_logger().warn('Map data is not available yet.')
            return None

        # Convert the world (x, y) coordinates into map grid coordinates (column, row)
        col = int((x - self.map_origin_x) / self.map_resolution)
        row = int((y - self.map_origin_y) / self.map_resolution)

        # Ensure the coordinates are within the map boundaries
        if col < 0 or col >= self.map_width or row < 0 or row >= self.map_height:
            self.get_logger().warn(f'Position ({x}, {y}) is outside the map boundaries.')
            return None

        # Convert the row/column to a 1D index in the map data array
        index = row * self.map_width + col
        cell_value = self.map_data[index]

        # Return whether the cell is free, occupied, or unknown
        if cell_value == 0:
            return 'free'
        elif cell_value == 100:
            return 'occupied'
        else:
            return 'unknown'


class NavToPoseClient(Node):
    def __init__(self):
        super().__init__('nav_to_pose_client')
        
        # Create an action client for the NavigateToPose action
        self._action_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # Create a subscription to the image topic (replace '/camera/image_raw' with your image topic)
        self.image_subscription = self.create_subscription(
            Image,
            '/camera/image_raw',  # Replace with your image topic
            self.image_callback,
            10)
        
        # Create a subscription to the AMCL pose topic (to get robot's position and orientation)
        self.pose_subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',  # AMCL pose topic, may vary depending on your setup
            self.pose_callback,
            10)

        # Initialize CvBridge for converting ROS images to OpenCV format
        self.bridge = CvBridge()

        # Directory to save images
        self.image_save_path = os.path.join(os.getcwd(), 'saved_images')
        if not os.path.exists(self.image_save_path):
            os.makedirs(self.image_save_path)

        # CSV file to store poses corresponding to each image
        self.pose_save_path = os.path.join(self.image_save_path, 'image_poses.csv')
        with open(self.pose_save_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Image', 'X', 'Y', 'Yaw (degrees)'])

        # Image count for naming saved images
        self.image_count = 0

        # Variables to store robot's current position and yaw angle
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

    def image_callback(self, msg):
        # Convert ROS Image message to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Generate a file name and save the image
        image_filename = f'image_{self.image_count:04d}.png'
        image_path = os.path.join(self.image_save_path, image_filename)
        cv2.imwrite(image_path, cv_image)

        # Save the robot's pose along with the image name
        with open(self.pose_save_path, 'a', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow([image_filename, self.robot_x, self.robot_y, self.robot_yaw])

        
        # Log the image saving process
        self.get_logger().info(f'Saved image {image_filename}')
        
        # Increment the image counter
        self.image_count += 1


    def pose_callback(self, msg):
        # Get the robot's current position (x, y)
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Extract the yaw angle (z-axis rotation) from the quaternion orientation
        orientation_q = msg.pose.pose.orientation
        _, _, yaw = tf_transformations.euler_from_quaternion([
            orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w
        ])

        # Convert yaw from radians to degrees
        self.robot_yaw = math.degrees(yaw)

        self.get_logger().info(f'X: {self.robot_x}, Y: {self.robot_y}, Yaw: {self.robot_yaw}')


    def send_goal(self, pose: PoseStamped):
        self.get_logger().info('Waiting for action server...')
        self._action_client.wait_for_server()

        # Create the goal message
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        # Send the goal
        self.get_logger().info('Sending goal to move to target pose...')
        self._goal_future = self._action_client.send_goal_async(goal_msg)
        self._goal_future.add_done_callback(self.goal_response_callback)


    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected.')
            return

        self.get_logger().info('Goal accepted.')
        self._result_future = goal_handle.get_result_async()
        self._result_future.add_done_callback(self.get_result_callback)


    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info('Goal reached with result: {0}'.format(result))

        rclpy.shutdown()
        

def main(args=None):
    rclpy.init(args=args)

    # Create a navigation client node
    nav_client = NavToPoseClient()

    # Create the node
    # map_checker = MapChecker()

    # # Wait for the map to be available
    # rclpy.spin_once(map_checker)

    # Define the target pose (replace these with actual pose values in your map frame)
    pose = PoseStamped()
    pose.header.frame_id = "map"  # Target frame (usually "map")
    pose.header.stamp = nav_client.get_clock().now().to_msg()

    # x, y = random.uniform()

    # Set position (x, y, z)
    pose.pose.position.x = 2.0
    pose.pose.position.y = 3.0
    pose.pose.position.z = 0.0

    # Set orientation (quaternion)
    # Assuming no rotation in the x, y plane (0 roll, 0 pitch), only yaw in z-axis
    yaw = math.radians(90)  # 90 degrees in radians
    pose.pose.orientation.z = math.sin(yaw / 2.0)
    pose.pose.orientation.w = math.cos(yaw / 2.0)

    # Send the goal pose
    nav_client.send_goal(pose)

    rclpy.spin(nav_client)


if __name__ == '__main__':
    main()
