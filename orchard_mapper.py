#!/usr/bin/env python3


import rospy
from sensor_msgs.msg import PointField, PointCloud2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
import struct
from std_msgs.msg import Header
import sys
import pandas as pd
from collections import defaultdict
from std_srvs.srv import Empty
from geometry_msgs.msg import Pose
import glob
import os
from ptcloud_visualization.srv import GiveNextPc
import cv2
import matplotlib.pyplot as plt
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import utm
import math
import csv
from natsort import natsorted
from yolo_detector import detect_boxes, convert_scale
from pocloud_processor import refresh, pc_process
from mapping_utils import do_transformation, get_distance, set_key, get_heading, create_box_list, create_near_points_list, remove_ground_points, create_coord_candidates, create_target_coords, track_instances, draw_map, create_gps_list


# foreground_tree_index: This number is to determin whether the detected tree is foreground or background, if below, then background tree.
foreground_tree_index = 0.188

# thresh: If the distance between two trees in consecutive images are below thresh, then they are two different tree
thresh = 0.955 
dist_btw_cam_gps = 0.7
camera_heading = 0
PC_MESSAGE=None


def main():
    # Start the node 'pc_publisher'.
    rospy.init_node('pc_publisher')

    csv_file = '/home/woo/Desktop/draw_orchard_map/gps_coordinates_cleaned.csv'
    vehicle_gps_x_list, vehicle_gps_y_list = create_gps_list(csv_file)


    path_to_files = "/home/woo/Desktop/plydata/processed_data"
    files_to_read = natsorted(os.listdir(path_to_files))

    # ender: ROS end indicator, t: image number, tree_id: tree_id: to initialize tree instance tracking, tree_dict_key: to initialize the tree dictionary key.
    ender = 0
    t = 0
    tree_id = 1
    tree_dict_key = 0
    publish_topic = '/published_pc'
    pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)
    
    tree_dict = dict()
    image_dict = dict()

    for a in files_to_read:

        # Calculate heading of the vehicle
        vehicle_heading = get_heading(t, vehicle_gps_y_list, vehicle_gps_x_list)

        b = os.path.join(path_to_files, a)

        # Read pointcloud
        pcd, PC_MESSAGE = refresh(b)
        rate = rospy.Rate(1)
        PC_MESSAGE.header.stamp = rospy.Time.now()
        pub.publish(PC_MESSAGE)

        # Separate RGBD data by RGB and Depth channel
        ori_xyz_pcd, ori_rgb_pcd, ori_bgr_img = pc_process(pcd)

        # bbox_bgr_img is an image with bbox on the detected trunks and boxes is list of lists with [x,y,w,h].
        bbox_bgr_img, boxes = detect_boxes(ori_bgr_img)


        # It is to check if the tree is detected.
        if len(boxes) != 0:
            # Create a list of box lists.
            bbox_points_list = create_box_list(boxes, ori_xyz_pcd)

            # Filter points by depth.
            near_bbox_points_x_list, near_bbox_points_y_list, near_bbox_points_z_list, foreground_ratio, num_of_z_points, num_of_filtered_z_points = create_near_points_list(bbox_points_list)
                    
            centers = []
            valid_bbox_points_x_lists = []
            valid_bbox_points_y_lists = []
            valid_bbox_points_z_lists = []


            for i in range(len(foreground_ratio)):
                # Check if the detected tree is a foreground tree of a background tree.
                if foreground_ratio[i] >= foreground_tree_index:

                    center, valid_bbox_points_x_list, valid_bbox_points_y_list, valid_bbox_points_z_list = remove_ground_points(i, boxes, near_bbox_points_x_list, near_bbox_points_y_list, near_bbox_points_z_list)

                    centers.append(center)                 
                    valid_bbox_points_x_lists.append(valid_bbox_points_x_list) 
                    valid_bbox_points_y_lists.append(valid_bbox_points_y_list) 
                    valid_bbox_points_z_lists.append(valid_bbox_points_z_list)

            # If the detected tree was determined as foreground tree, then estimate the tree location in the world frame.     
            if len(valid_bbox_points_y_lists) != 0:

                txw = []
                tyw = []

                m = len(valid_bbox_points_y_lists)
                for i in range(m):
                    # Get all possible x,y,z coordinates of the tree in camera frame. 
                    x_coords, y_coords, z_coords = create_coord_candidates(valid_bbox_points_x_lists, valid_bbox_points_y_lists, valid_bbox_points_z_lists, i)

                    if len(x_coords) != 0 and len(y_coords) != 0 and len(z_coords) != 0:

                        # Estimate the tree's coordinate in camera frame by averaging the possible coordinates.
                        target_x, target_y, target_z = create_target_coords(x_coords, y_coords, z_coords)

                        # Calculate distance between tree and camera(or vehicle)
                        dist_to_tree = np.sqrt(target_x**2 + target_z**2)

                        # Transform tree pose in camera frame to the world frame.
                        tx_w, ty_w = do_transformation(target_x, target_z, vehicle_gps_x_list[t], vehicle_gps_y_list[t], vehicle_heading, camera_heading, dist_btw_cam_gps)


                        txw.append(tx_w)
                        tyw.append(ty_w)
                                         
                # Track instance of trees to avoid double counting.                                                                          
                tree_id, tree_dict, image_dict = track_instances(t, tree_id, centers, image_dict, tree_dict, tree_dict_key, txw, tyw, m, thresh)

                print("tree id : ", tree_id)
                tree_dict_key += 1


        t+=1

    if ender == 0:
        rospy.signal_shutdown("Read all the ply files!")


    draw_map(tree_dict)


if __name__ == "__main__":

    main()


