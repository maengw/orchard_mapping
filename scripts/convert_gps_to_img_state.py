#!/usr/bin/env python

from natsort import natsorted
import matplotlib.pyplot as plt
import cv2
from tree_detector import detect_trees
from depth_utils import separate_data, create_pcd_box, check_foreground_tree
import rospy
from map_utils import create_gps_list
import os
from sensor_msgs.msg import PointField, PointCloud2
from pc_utils import read_ply
import csv
from pandas import *
import math


from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_relation():
    """manually go over images one by one and save center of detected tree and corresponding gps data of vehicle. 
    it was done initially before trying linear regression
    as regression would be sufficient for this. only did up to 85 images and stopped; thus the function is unnecessary
    """
    

    csv_file = '/home/woo/catkin_ws/src/orchard_mapping/data/gps_coordinates_cleaned.csv'

    vehicle_gps_x, vehicle_gps_y = create_gps_list(csv_file)

    vehicle_gps_x, vehicle_gps_y = vehicle_gps_x[:85], vehicle_gps_y[:85]

    path_to_files = "/home/woo/catkin_ws/src/orchard_mapping/data/sample_plydata"

    files_to_read = natsorted(os.listdir(path_to_files))
    files_to_read = files_to_read[:85]

    rospy.init_node('gps_to_img_states_converter')
    
    end_ros = True

    publish_topic = '/published_pc'

    pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)

    tree_id = []
    gps_x = []
    gps_y = []
    centers = []

    filenum = 0
    for filename in files_to_read:
        print("file number : ", filenum)
        ply_file = os.path.join(path_to_files, filename)

        pts = read_ply(ply_file)

        xyz_pcd, bgr_img = separate_data(pts)

        b_box_img, b_boxes = detect_trees(bgr_img)

        # cv2.imshow('window', b_box_img)
        # cv2.waitKey(1)

        if len(b_boxes) != 0:
            for box in b_boxes:
                pcd_box = create_pcd_box(xyz_pcd, box)

                center = ( (box[0]+box[2])/2, 240)
                print("The center of the box you are looking at is ", center)
                print("Checking if it is foreground or background")
                if check_foreground_tree(pcd_box):
                    message = input("What is the tree id ?\n")
                    tree_id.append(message)
                    gps_x.append(vehicle_gps_x[filenum])
                    gps_y.append(vehicle_gps_y[filenum])
                    centers.append(center)

        filenum += 1
    cv2.destroyAllWindows()

    if end_ros == True:
        rospy.signal_shutdown("Read all the ply files!")



    header = ['tree_id', 'gps_x', 'gps_y', 'tree_center']

    with open("tree_gps_img_relation.csv", "w") as f:
        writer = csv.writer(f)

        writer.writerow(header)
        for i in range(len(tree_id)):
            writer.writerow([tree_id[i], gps_x[i], gps_y[i], centers[i]])
  

def regression():
    """linear regression to find relation between the amount of tree moved in image 
    with vehicle's travel distance in world frame"""

    csv_file = '/home/woo/catkin_ws/src/orchard_mapping/data/tree_gps_img_relation_dist2.csv'

 
    # reading CSV file
    data = read_csv(csv_file)
 
    # converting column data to list
    dist_diff = data['dist_diff'].tolist()
    pixel_diff = data['pixel_diff'].tolist()
    diff_d = [[x] for x in dist_diff if math.isnan(x) == False]
    diff_p = [[x] for x in pixel_diff if math.isnan(x) == False]

    line_fitter = LinearRegression()
    line_fitter.fit(diff_d, diff_p)
    print(line_fitter.predict([[70]]))
    print(line_fitter.coef_) # slope = -631.32630251
    print(line_fitter.intercept_) # y_intercept = -5.0538143

    y = -631.32630251*70 - 5.0538143
    print(y)


    plt.plot(diff_d, diff_p, 'o')
    plt.plot(diff_d, line_fitter.predict(diff_d))
    plt.show()
    #print("average distance diff: ", sum(diff_d)/len(diff_d))
    #print("average pixel diff: ", sum(diff_p)/len(diff_p))
    #plt.scatter(diff_d, diff_p)
    #plt.show()

 
def estimate_pixel_displacement(x):
    y = -631.32630251*x - 5.0538143
    return y

if __name__ == "__main__":

    regression()
