#!/usr/bin/env python


import numpy as np
import pandas as pd
import utm


def initial_heading(y1, x1, y2, x2):
    heading = np.arctan2(y1 - 2*(y2 - y1), x1 - 2*(x2 - x1))
    return heading


def calculate_heading(y1, x1, y2, x2):
    heading = np.arctan2(y2 - y1, x2 - x1)
    return heading


def calculate_transformation(cam_tree_x, cam_tree_y, gps_x, gps_y, vehicle_heading, camera_heading, dist_btw_cam_gps):
    """transforms estimated tree's location in camera frame to world frame
    @param cam_tree_x - tree's x position with respect to the camera frame
    @param cam_tree_y - tree's y position with respect to the camera frame
    @param gps_x - vehicle's gps x in world frame
    @param gps_y - vehicle's gps y in world frame
    @param vehicle_heading - heading of the vehicle in world frame
    @param camera_heading - heading of the camera with respect to the vehicle.
    @param dist_btw_cam_gps - distance between the camera and RTK-GPS unit.
    @returns tree's x and y in world frame.
    """  

    T_wv = np.array([[np.cos(vehicle_heading), -np.sin(vehicle_heading), gps_x], [np.sin(vehicle_heading), np.cos(vehicle_heading), gps_y], [0, 0, 1]])

                                                           
    T_vc = np.array([[np.cos(camera_heading), -np.sin(camera_heading), dist_btw_cam_gps], [np.sin(camera_heading), np.cos(camera_heading), 0], [0, 0, 1]])

    tree_c = np.array([[cam_tree_x], [cam_tree_y], [1]])

    tree_w_xy = np.dot(np.dot(T_wv, T_vc), tree_c)


    # return tree_w_xy[0][0], tree_w_xy[1][0]
    return tree_w_xy[0,0], tree_w_xy[1,0]

def calculate_pythagorean(x, y):
    return np.sqrt(x**2 + y**2)

def calculate_dist(x_1, y_1, x_2, y_2):
    return np.sqrt((x_2-x_1)**2 +(y_2-y_1)**2)
    

def create_gps_list(csv_file):
    """creates a list of x and y of vehicle
    @param csv_file - vehicle gps csv file with the whole path to directory
    @returns gps x and y of the vehicle
    """
    column_names = ['Long', 'Lat']
    df = pd.read_csv(csv_file)
    longitudes = df.Long.tolist()
    latitudes = df.Lat.tolist()

    vehicle_gps_x_list = []
    vehicle_gps_y_list = []
    for i in range(len(longitudes)):
        x, y, _, _ = utm.from_latlon(latitudes[i], longitudes[i])
        vehicle_gps_x_list.append(x)
        vehicle_gps_y_list.append(y)

    return vehicle_gps_x_list, vehicle_gps_y_list


def create_tree_dict(dictionary, key, value):
    # Keys of dictionary are tree id and the values are corresponding potential coordinates.
    if key not in dictionary:
        dictionary[key] = [value]

    else:
        dictionary[key].append(value)

