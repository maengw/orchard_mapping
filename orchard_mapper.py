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
from yolo_detector import create_bbox, convert
from pocloud_processor import refresh, pc_process
from mapping_utils import do_transformation, get_distance, set_key, get_heading, create_box_list, create_valid_points_list, remove_points, create_coord_candidates, create_target_coords, track_instances, draw_map



def main():
    # start the node 'pc_publisher'.
    rospy.init_node('pc_publisher')

    csv_file = '/home/woo/Desktop/draw_orchard_map/gps_coordinates_cleaned.csv'
    column_names = ['Long', 'Lat']
    df = pd.read_csv(csv_file)
    longitudes = df.Long.tolist()
    latitudes = df.Lat.tolist()

    vehicle_gps_x_list = []
    vehicle_gps_y_list = []
    for qq in range(len(longitudes)):

        x, y, _, _ = utm.from_latlon(latitudes[qq], longitudes[qq])
        vehicle_gps_x_list.append(x)
        vehicle_gps_y_list.append(y)

    # path_to_files = "/home/woo/Desktop/sample_plydata"
    path_to_files = "/home/woo/Desktop/plydata/processed_data"
    # files_to_read = sorted(os.listdir(path_to_files))
    files_to_read = natsorted(os.listdir(path_to_files))


    ender = 0
    t = 0
    tree_id = 1
    publish_topic = '/published_pc'

    tree_dict = dict()
    image_dict = dict()

    tt = 0

    thresh = 1.0 # 0.955

    for a in files_to_read:

        # print("image number: ", t)


        # code to calculate heading
        vehicle_heading = get_heading(t, vehicle_gps_y_list, vehicle_gps_x_list)

        b = os.path.join(path_to_files, a)
        pcd, PC_MESSAGE = refresh(b)

        pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)
        rate = rospy.Rate(1)

        PC_MESSAGE.header.stamp = rospy.Time.now()
        pub.publish(PC_MESSAGE)

        ori_xyz_pcd, ori_rgb_pcd, ori_bgr_img = pc_process(pcd)

        # bbox_bgr_img is an image with bbox on the detected trunks and boxes is list of lists with [x,y,w,h].
        bbox_bgr_img, boxes = create_bbox(ori_bgr_img)



        if len(boxes) != 0:
            bbox_points_list = create_box_list(boxes, ori_xyz_pcd)

            valid_bbox_points_x_list, valid_bbox_points_y_list, valid_bbox_points_z_list, num_of_z_points, num_of_filtered_z_points = create_valid_points_list(bbox_points_list)
            
            centers = []
            valid_bbox_points_x_array = []
            valid_bbox_points_y_array = []
            valid_bbox_points_z_array = []

            for i in range(len(num_of_z_points)):
                foreground_ratio = num_of_filtered_z_points[i] / num_of_z_points[i]

                if foreground_ratio >= 0.188:


                    center, valid_bbox_points_x_list, valid_bbox_points_y_list, valid_bbox_points_z_list = remove_points(i, boxes, valid_bbox_points_x_list, valid_bbox_points_y_list, valid_bbox_points_z_list)


                    centers.append(center)                 
                    valid_bbox_points_x_array.append(valid_bbox_points_x_list) 
                    valid_bbox_points_y_array.append(valid_bbox_points_y_list) 
                    valid_bbox_points_z_array.append(valid_bbox_points_z_list)
     

            valid_bbox_points_y_nparray = np.array(valid_bbox_points_y_array)


            if len(valid_bbox_points_y_nparray) != 0:


                txw = []
                tyw = []

                m = len(valid_bbox_points_y_nparray)
                #for i in range(len(valid_bbox_points_y_nparray)):
                for i in range(m):
                    x_coords, y_coords, z_coords = create_coord_candidates(valid_bbox_points_y_nparray, valid_bbox_points_x_array, valid_bbox_points_y_array, valid_bbox_points_z_array, i)

                    if len(x_coords) != 0 and len(y_coords) != 0 and len(z_coords) != 0:


                        target_x, target_y, target_z = create_target_coords(x_coords, y_coords, z_coords)

                        # Calculate distance to the tree
                        dist_to_tree = np.sqrt(target_x**2 + target_z**2)
                                                                                                                                       # - np.pi/2
                        tx_w, ty_w = do_transformation(target_x, target_z, vehicle_gps_x_list[t], vehicle_gps_y_list[t], vehicle_heading           , 0)# camera_heading)


                        txw.append(tx_w)
                        tyw.append(ty_w)
                                                                                                                      # tt = key
                tree_id, tree_dict, image_dict, tree_dict = track_instances(t, tree_id, centers, image_dict, tree_dict, tt, txw, tyw, m)

                print("tree id : ", tree_id)
                tt += 1


        t+=1
        # print(tree_dict)


    if ender == 0:
        rospy.signal_shutdown("Read all the ply files!")


    draw_map(tree_dict)


if __name__ == "__main__":

    main()

