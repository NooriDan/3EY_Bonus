#!/usr/bin/env python

import  rospy
from    sensor_msgs.msg import LaserScan, Image
from    cv_bridge import CvBridge
import  cv2
import  numpy as np
import  pyrealsense2 as pr

class LidarDepthFusionNode:
    def __init__(self):
        rospy.init_node('lidar_depth_fusion_node', anonymous=True)
        self.bridge = CvBridge()

        # Subscribe to LiDAR data
        self.lidar_sub = rospy.Subscriber('/lidar_topic', LaserScan, self.lidar_callback)

        # Subscribe to depth data from RealSense camera
        self.depth_sub = rospy.Subscriber('/depth_camera_topic', Image, self.depth_callback)

        # Publisher for corrected LiDAR data
        self.lidar_corrected_pub = rospy.Publisher('/corrected_lidar_topic', LaserScan, queue_size=10)

        # Parameters
        self.camera_fov = 60  # Change as per your camera's FOV
        self.lidar_fov = 360  # LiDAR's FOV
        self.max_depth_diff = 0.2  # Maximum difference to consider for depth correction

        # Initialize LiDAR and depth data
        self.lidar_data = None
        self.depth_data = None

    def lidar_callback(self, data):
        self.lidar_data = data

    def depth_callback(self, data):
        try:
            # Convert Image to numpy array
            depth_image = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            self.depth_data = depth_image
        except Exception as e:
            print(e)

    def process_data(self):
        if self.lidar_data is not None and self.depth_data is not None:
            lidar_ranges = self.lidar_data.ranges
            lidar_angles = np.linspace(self.lidar_data.angle_min, self.lidar_data.angle_max, len(lidar_ranges))

            # Filter depth data within camera FOV
            camera_fov = np.deg2rad(self.camera_fov)
            fov_mask = np.logical_and(lidar_angles > -camera_fov / 2, lidar_angles < camera_fov / 2)
            lidar_ranges_filtered = np.array(lidar_ranges)[fov_mask]

            corrected_ranges = []

            for i, angle in enumerate(lidar_angles):
                if not fov_mask[i]:
                    corrected_ranges.append(lidar_ranges[i])
                    continue

                # Calculate corresponding pixel column in the depth image
                col = int((angle - self.lidar_data.angle_min) / self.lidar_data.angle_increment)
                if col < 0 or col >= self.depth_data.shape[1]:
                    corrected_ranges.append(lidar_ranges[i])
                    continue

                # Find corresponding depth value from the depth image
                depth = self.depth_data[:, col]
                valid_depths = depth[depth > 0]  # Filter out invalid depths (zeros)
                if len(valid_depths) == 0:
                    corrected_ranges.append(lidar_ranges[i])
                    continue

                closest_depth = np.min(valid_depths)
                depth_diff = closest_depth - lidar_ranges[i]

                # Correct LiDAR range if depth difference is within threshold
                if abs(depth_diff) < self.max_depth_diff:
                    corrected_ranges.append(lidar_ranges[i] + depth_diff)
                else:
                    corrected_ranges.append(lidar_ranges[i])

            # Publish corrected LiDAR data
            corrected_scan = LaserScan()
            corrected_scan.header = self.lidar_data.header
            corrected_scan.angle_min = self.lidar_data.angle_min
            corrected_scan.angle_max = self.lidar_data.angle_max
            corrected_scan.angle_increment = self.lidar_data.angle_increment
            corrected_scan.time_increment = self.lidar_data.time_increment
            corrected_scan.scan_time = self.lidar_data.scan_time
            corrected_scan.range_min = self.lidar_data.range_min
            corrected_scan.range_max = self.lidar_data.range_max
            corrected_scan.ranges = corrected_ranges
            self.lidar_corrected_pub.publish(corrected_scan)

    def run(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.process_data()
            rate.sleep()

if __name__ == '__main__':
    try:
        node = LidarDepthFusionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
