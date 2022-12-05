#!/usr/bin/env python


from natsort import natsorted
import rospy
from map_utils import create_gps_list
import os
from sensor_msgs.msg import PointCloud2
from map_utils import initial_heading, calculate_heading, calculate_transformation, create_tree_dict
from pc_utils import read_ply
from depth_utils import separate_data, create_pcd_box, check_foreground_tree, filter_pts_by_dist, filter_ground_points, estimate_tree_location
from tree_detector import detect_trees
import cv2
import numpy as np
import math
from convert_gps_to_img_state import estimate_pixel_displacement
import matplotlib.pyplot as plt
import csv


def main():
   """draw bounding box on trees and write tree id by the box to show object tracking"""  

    # distance between installed camera and a gps device on a vehicle
    dist_btw_cam_gps = 0.7
    # initial camera heading w/r to the gps device
    camera_heading = 0
    # a .csv file that has gps data of a vehicle while collecting images in the orchard
    csv_file = '/home/woo/catkin_ws/src/orchard_mapping/data/gps_coordinates_cleaned.csv'

    # initiate ros node 'mapper'
    rospy.init_node('mapper')

    # create a list of vehicle gps data in world frame
    gps_x, gps_y = create_gps_list(csv_file)

    #path_to_files = rospy.get_param('path_to_ply')
    #files_to_read = natsorted(os.listdir(path_to_files))

    # .ply files(pointcloud data collected in the orchard)
    path_to_files = '/home/woo/catkin_ws/src/orchard_mapping/data/sample_plydata'
    # print("number of files:", len(path_to_files))
    # files_to_read = natsorted(os.listdir(path_to_files))[:85]
    files_to_read = natsorted(os.listdir(path_to_files))
    # if want to use rosparam - not necessary for this time
    #path_to_files = rospy.get_param('path_to_ply')

    # indicator to end ros
    end_ros = True


    publish_topic = '/published_pc'

    pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)

    # variable to track the loop
    feed_num = 0

    # tree instance tracker track id 
    track_id = 0

    # dictionary to save tree id and its corresponding coordinates in world
    tree_dict = {}
    tree_coords = {}


    # a list to store center of bounding box of the previous frame in pixel coordinates
    center_pts_prev_frame = []
    # a list to store tree coordinates that was detected in a previous frame in gps(lat/lon) coordinates
    trees_prev_frame = []
    # a list that has above two data
    center_pts_n_tree_gps_prev_frame = []

    # dictionary to track tree instances
    tracking_objects = {}

    # loop through .ply files
    for filename in files_to_read:
        print("CHECK feed_num: ", feed_num)

        # Calculate heading of the vehicle
        if feed_num == 0:
            x1, y1, x2, y2 = gps_x[0], gps_y[0], gps_x[1], gps_y[1]
            heading = initial_heading(y1, x1, y2, x2)
        else:
            x1, y1, x2, y2 = gps_x[feed_num-1], gps_y[feed_num-1], gps_x[feed_num], gps_y[feed_num]
            heading = calculate_heading(y1, x1, y2, x2)


        ply_file = os.path.join(path_to_files, filename)

        # read a ply file
        pts = read_ply(ply_file)
        # separate x,y,z,r,g,b, data into two
        xyz_pcd, bgr_img = separate_data(pts)

        # b_box_img, b_boxes = detect_trees(bgr_img)
        _, b_boxes = detect_trees(bgr_img)

        # a list to store center points of bounding box of a current frame in pixel coordinate system
        center_pts_cur_frame = []
        # a list to store trees' coordinates in gps(lat/lon) coordinate system that is detected in current frame
        trees_cur_frame = []
        # merged above two lists
        center_pts_n_tree_gps_cur_frame = []

        # if tree is detected
        if len(b_boxes) != 0:
            valid_b_boxes = []
            valid_pcd_boxes = []

            for box in b_boxes:
                pcd_box = create_pcd_box(xyz_pcd, box)
                # check if the detected tree is a foreground tree or a background tree; we only want the trees in the foreground
                if check_foreground_tree(pcd_box):
                    valid_b_boxes.append(box)
                    valid_pcd_boxes.append(pcd_box)
                    # y value of the center will be fixed to 240, because the vehicle moved only in horizontal
                    center = [int((box[0]+box[2])/2), 240]
                    # save the center of bounding box that is drawn on the detected foreground tree
                    center_pts_cur_frame.append(center)

            # filter pointcloud by 1. distance; removed points that are too close( < 0.3m ) or too far( > 1.5m)
            #                      2. ground points; removed points that are close to the ground because there may be big obstacles on the ground
            for val_box in valid_pcd_boxes:
                x_pts, y_pts, z_pts = filter_pts_by_dist(val_box)
                x_pts, y_pts, z_pts = filter_ground_points(x_pts, y_pts, z_pts)
                # estimate tree location by calculating average of the remaining points with respect to the camera frame
                x, y, z = estimate_tree_location(x_pts, y_pts, z_pts)
                # transform tree location into world frame; gps(lat/lon) coordinate system
                tree_x, tree_y = calculate_transformation(x, z, gps_x[feed_num], gps_y[feed_num], heading, camera_heading=0, dist_btw_cam_gps=0.7)
                trees_cur_frame.append([tree_x, tree_y])

            # merge current center points and tree location 
            for i in range(len(center_pts_cur_frame)):
                center_pts_n_tree_gps_cur_frame.append(center_pts_cur_frame[i] + trees_cur_frame[i])
            
            # Object(tree instance) tracking
            # this will run only at first
            if feed_num  < 1:
                for pt in center_pts_n_tree_gps_cur_frame:
                    for pt2 in center_pts_n_tree_gps_prev_frame:
                        # distance between center of previous frame and current frame in pixel frame
                        pix_dist = pt2[0] - pt[0] 
                        # distance between the trees in previous frame and current frame
                        gps_dist = math.hypot(pt2[2] - pt[2], pt2[3] - pt[3])

                        # vehicle moved only forward and never went backwards when collecting the data
                        # if ((pix_dist <= 200) and (gps_dist <= 1.5)):
                        # thus, the difference between to frames should always be > 0 , if same instance
                        if ((pix_dist > 0) and (gps_dist <= 1.5)):
                            tracking_objects[track_id] = pt
                            track_id += 1
            else:
                # make copy of dict and list in order to safely loop through and process data safely
                tracking_objects_copy = tracking_objects.copy()
                center_pts_n_tree_gps_cur_frame_copy = center_pts_n_tree_gps_cur_frame.copy()

                # used vehicle gps data and center points displacement for linear regression 
                # and derived equation that shows relation between pixel displacement and vehicle movement
                # pix_dist_thresh = abs(estimate_pixel_displacement(math.hypot(x2-x1, y2-y1)))

                for object_id, pt2 in tracking_objects_copy.items():
                    object_exists = False

                    for pt in center_pts_n_tree_gps_cur_frame_copy:

                        pix_dist = pt2[0] - pt[0]
                        gps_dist = math.hypot(pt2[2] - pt[2], pt2[3] - pt[3])

                        # update object position
                        # if ((pix_dist <= pix_dist_thresh*1.2) and (gps_dist <= 1.5)):
                        if ((pix_dist > 0) and (gps_dist <= 1.5)):
                            tracking_objects[object_id] = pt
                            object_exists = True
                            if pt in center_pts_n_tree_gps_cur_frame:
                                center_pts_n_tree_gps_cur_frame.remove(pt)
                                continue

                    if not object_exists:
                        tracking_objects.pop(object_id)

                for pt in center_pts_n_tree_gps_cur_frame:
                    tracking_objects[track_id] = pt
                    track_id += 1


            for object_id, pt in tracking_objects.items():

                cv2.circle(bgr_img, (pt[0:2]), 5, (0,0,255), -1)
                cv2.putText(bgr_img, str(object_id), (pt[0], pt[1]-7), 0, 1, (0, 0, 255), 2)
         
                create_tree_dict(tree_dict, object_id, pt[2:])

            center_pts_prev_frame = center_pts_cur_frame.copy()
            trees_prev_frame = trees_cur_frame.copy()

            center_pts_n_tree_gps_prev_frame = center_pts_n_tree_gps_cur_frame.copy()

        # cv2.imwrite('/home/woo/catkin_ws/src/orchard_mapping/result1/detection%s.png' %feed_num, bgr_img)
        cv2.imshow('fig', bgr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        feed_num += 1

    if end_ros == True:
        rospy.signal_shutdown("Read all the ply files!")


if __name__ == "__main__":

    main()


