#!/usr/bin/env python


import numpy as np
import cv2



def convert_scale(img, target_type_min, target_type_max, target_type):
    """convert data type and rescale the data
    @param img - rgb image
    @param target_type_min - target minimum value
    @param target_type_max - target maximum value
    @param target_type - target data type
    @returns a new image with changed data type and new scale
    """

    imin = 0
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def separate_data(pcd):
    """Separates 6d depth image into two 3d arrays; BGR image and XYZ array
    @param pcd - pointcloud created from read_ply() in pc_utils.py [x,y,z,r,g,b]
    @return 3d BGR image and 3d XYZ image"""

    xyz_pcd = pcd[:, 0:3]
    rgb_pcd = pcd[:, 3:6]

    rgb_img = np.reshape(rgb_pcd, (480, 848, 3)) / 255
    rgb_img = convert_scale(rgb_img, 0, 255, np.uint8)
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    xyz_pcd = np.reshape(xyz_pcd, (480, 848, 3))

    return xyz_pcd, bgr_img


def create_pcd_box(xyz_pcd, boxes):
    """creates a list of pcd array in bounding boxes
    @param xyz_pcd - 3d xyz array from separate_data()
    @param boxes - a list of bounding boxes' edges from detect_trees in tree_detector.py 
    @returns a list of pcd of detected tree(bounding boxes)
    """
    pcd_boxes = []
    for box in boxes:
        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]
        pcd_box = xyz_pcd[y1:y2, x1:x2, :]
        pcd_boxes.append(pcd_box)
        return pcd_boxes

    # line 59-64 is only used when running convert_gps_to_img_state.py; use line 48-56 elsewhere.
    # x1 = boxes[0]
    # x2 = boxes[2]
    # y1 = boxes[1]
    # y2 = boxes[3]
    # pcd_box = xyz_pcd[y1:y2, x1:x2, :]
    return pcd_box

def check_foreground_tree(pcd_box):
    """checks wether the detected tree is a foreground tree or a backgroud tree
    @param pcd_box - a  list of a pcd bounding box from create_pcd_box()
    @ returns True or False"""

    pts_z = pcd_box[:, :, 2]
    non_nan_z_pts = pts_z[~np.isnan(pts_z)]


    num_z_pts = len(non_nan_z_pts)

    # print("number of no nan z points: ", num_z_pts)

    filtered_z_pts = non_nan_z_pts[(non_nan_z_pts <= 1.5)]
    num_fil_z_pts = len(filtered_z_pts)

    # print("number of filtered z points: ", num_fil_z_pts)

    ratio = (num_fil_z_pts / num_z_pts)
    # print(ratio)

    if ratio >= 0.7:
        # print("It is a FOREGROUND tree. The ratio is ",ratio)
        return True
    # print("It is a BACKGROUND tree. The ratio is ",ratio)
    return False

    

#     nonzeros = []


def filter_pts_by_dist(pcd_box, foreground=True):
    """filters points that are far or too close; this will filter points in the background and
    points corresponding to near obstacles that are in the bounding box. 1.5m > and 0.2m < will be filtered
    @param pcd_boxes - a pcd bounding box from create_pcd_box()
    @returns list of x,y,z points that are filtered by distance
    """

    if foreground == True:
        pts_x = pcd_box[:, :, 0].flatten()
        pts_y = pcd_box[:, :, 1].flatten()
        pts_z = pcd_box[:, :, 2].flatten()

        index_of_unwanted_pts = []
        for i in range(len(pts_y)):
            if pts_z[i] > 1.5 or pts_z[i] < 0.2:
                index_of_unwanted_pts.append(i)

        index_of_unwanted_pts.sort(reverse=True)


#         for idx in index_of_unwanted_pts:
#             pts_x.pop(idx)
#             pts_y.pop(idx)
#            pts_z.pop(idx)

        new_pts_x = np.delete(pts_x, index_of_unwanted_pts)
        new_pts_y = np.delete(pts_y, index_of_unwanted_pts)
        new_pts_z = np.delete(pts_z, index_of_unwanted_pts)


        return new_pts_x, new_pts_y, new_pts_z


def filter_ground_points(pts_x, pts_y, pts_z):
    """filters points that are near ground and NaN points
    @param x_pts, y_pts, z_pts - a list of points that are already filtered by distance from filter_pts_by_dist()
    @returns points that could be used for tree location estimation"""
    
    index_of_ground_pts = []

    for i in range(len(pts_y)):
        if pts_y[i] > 0.3:
            index_of_ground_pts.append(i)

    index_of_ground_pts.sort(reverse=True)

#     for idx in index_of_ground_pts:
#         pts_x.pop(idx)
#         pts_y.pop(idx)
#         pts_z.pop(idx)


    new_pts_x = np.delete(pts_x, index_of_ground_pts)
    new_pts_y = np.delete(pts_y, index_of_ground_pts)
    new_pts_z = np.delete(pts_z, index_of_ground_pts)



    valid_x_pts = new_pts_x[~np.isnan(new_pts_x)]
    valid_y_pts = new_pts_y[~np.isnan(new_pts_y)]
    valid_z_pts = new_pts_z[~np.isnan(new_pts_z)]


    return valid_x_pts, valid_y_pts, valid_z_pts




def estimate_tree_location(valid_x_pts, valid_y_pts, valid_z_pts):
    """estimate the location of tree with respect to the camera frame
    @param valid_pts - points that are completely filtered from previous filter functions
    @return estimate of tree location"""

    x = sum(valid_x_pts) / len(valid_x_pts)
    y = sum(valid_y_pts) / len(valid_y_pts)
    z = sum(valid_z_pts) / len(valid_z_pts)

    return x, y, z
