#!/usr/bin/env python

import rospy
from natsort import natsorted
import os
from sensor_msgs.msg import PointCloud2
# from map_utils import initial_heading, calculate_heading, calculate_transformation, create_tree_dict
from pc_utils import refresh, read_ply
from depth_utils import separate_data, create_pcd_box, check_foreground_tree, filter_pts_by_dist, filter_ground_points, estimate_tree_location
# from tree_detector import detect_trees
import cv2
import numpy as np
# import math
# from convert_gps_to_img_state import estimate_pixel_displacement
import matplotlib.pyplot as plt
import csv
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point

from orchard_mapping.srv import GiveNextPc


net = cv2.dnn.readNet('/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_training_1800.weights', '/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_testing.cfg')
classes = ['trunk']


    """visualize the positiion of detected tree by publishing marker on it.
    it is hard-coded as it is just to confirm that position estimation works.
    may need to work on refactoring the code in the future if needed."""

def bbox_creator(img):


	layer_names = net.getLayerNames()
	output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
	colors = np.random.uniform(0, 255, size=(len(classes), 3))

	height, width, _ = img.shape

	blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

	net.setInput(blob)

	output_layers_names = net.getUnconnectedOutLayersNames()
	layerOutputs = net.forward(output_layers_names)

	boxes = []
	confidences = []
	class_ids = []

	for output in layerOutputs:
		for detection in output:
		    scores = detection[5:]
		    class_id = np.argmax(scores)
		    confidence = scores[class_id]
		    if confidence > 0.5:
			    center_x = int(detection[0] * width)
			    center_y = int(detection[1] * height)
			    w = int(detection[2] * width)
			    h = int(detection[3] * height)

			    x = int(center_x - w/2)
			    y = int(center_y - h/2)

			    boxes.append([x, y, w, h])
			    confidences.append(float(confidence))
			    class_ids.append(class_id)

	indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

	font = cv2.FONT_HERSHEY_PLAIN
	colors = np.random.uniform(0, 255, size=(len(boxes), 3))


	trunk_array = []
	trunk_img_array = []


	if len(indexes) != 0:

		for i in indexes.flatten():
			cpimg = np.zeros((480,848,3))
			cpimg[:, :, :] = 255 # this will change to NaN later on
			x, y, w, h = boxes[i]
			label = str(classes[class_ids[i]])
			confidence = str(round(confidences[i], 2))
			color = colors[i]

			if x < 0:
			   x1 = 848
			   x2 = x + w
			else:
			   x1 = x
			   x2 = x + w

			if y < 0:
				y1 = 0
				y2 = 480
			else:
				y1 = y
				y2 = y + h

			roi = img[y1:y2, x1:x2, :]
			trunk_array.append(roi)

			cpimg[y1:y2, x1:x2] = roi
			trunk_img_array.append(cpimg)
		summ = np.zeros(np.shape(img))
		for i in range(len(trunk_img_array)):
			summ = summ + trunk_img_array[i]
		whole_img = summ - 255 * (len(trunk_img_array) - 1)

		cpimg2 = convert(whole_img, 0, 255, np.uint8)
		return cpimg2, trunk_img_array
	return img, trunk_img_array


def convert(img, target_type_min, target_type_max, target_type):
	imin = 0 # img.min()
	imax = img.max()

	a = (target_type_max - target_type_min) / (imax - imin)
	b = target_type_max - a * imax
	new_img = (a * img + b).astype(target_type)
	return new_img


def get_xyz(points):

    dim = len(points[0])
    if dim == 3:
        colors = False
    elif dim == 6:
        colors = True
    else:
        raise ValueError('Unsure how to interpret dimension {} input'.format(dim))

    if isinstance(points, np.ndarray):
        points = points.tolist()

    x_array = np.array([])
    y_array = np.array([])
    z_array = np.array([])

    for row in points:
        xyz = row[:3]
        x, y, z = row[:3]
        r, g, b = row[3:]
        r = int(r)
        g = int(g)
        b = int(b)
        a = 255
        if r+g+b == 765 or z > 1.4 or y > 0.3:
            pass
        else:
            rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            xyz.append(rgba)
            # formatted_points.append(xyz)
            if np.isnan(x) or np.isnan(y) or np.isnan(z) :
                pass
            else:

                x_array = np.append(x_array, x)
                y_array = np.append(y_array, y)
                z_array = np.append(z_array, z)

        
    argsorted_y_array = np.argsort(y_array)[::-1]
    argsorted_y_array = argsorted_y_array[:int(len(y_array)/10)]

    
    x_coord_cand = []
    y_coord_cand = []
    z_coord_cand = []

    for i in argsorted_y_array:
        x_coord_cand.append(x_array[i])
        y_coord_cand.append(y_array[i])
        z_coord_cand.append(z_array[i])


    x_coord = sum(x_coord_cand) / len(x_coord_cand)
    y_coord = sum(y_coord_cand) / len(y_coord_cand)
    z_coord = sum(z_coord_cand) / len(z_coord_cand)


    return [x_coord, y_coord, z_coord]


def rescale(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def create_newpoints(plyfile_to_read):

    pts = read_ply(plyfile_to_read)
    

    xyz_pcd = pts[:, 0:3]
    bgr_img = pts[:, 3:6]


    bgr_img = np.reshape(bgr_img, (480, 848, 3)) / 255
    bgr_img = rescale(bgr_img, 0, 255, np.uint8)

    b_box_img, b_boxes = bbox_creator(bgr_img)

    bbox_bgr_pcd = np.reshape(b_box_img, (407040, 3))
    return xyz_pcd, bbox_bgr_pcd, b_boxes, b_box_img


def visualize_target():
    """visualize pointcloud of the bounding box(detected tree) and visualize estimated position of the tree
    by publishing red circle marker on the target coordinate"""
    
    rospy.init_node('pc_visualizer')

    path_to_files = rospy.get_param('path_to_ply')
    # files_to_read = natsorted(os.listdir(path_to_files))
    files_to_read = os.listdir(path_to_files)
    rospy.wait_for_service("/give_next_pc")
    
    marker_array = MarkerArray()
    marker_array_publisher = rospy.Publisher('/marker_array', MarkerArray, queue_size=10, latch=True)



    publish_topic = '/published_pc'
    pc_publisher = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)
    pub_rate = 10
    rate = rospy.Rate(pub_rate)



    marker = Marker()
    marker.header.frame_id = "map"
    marker.header.stamp = rospy.get_rostime()

    marker.ns = ''
    marker.type = Marker.SPHERE
    marker.action = Marker.ADD

    marker.pose.orientation.x = 0.0
    marker.pose.orientation.y = 0.0
    marker.pose.orientation.z = 0.0
    marker.pose.orientation.w = 1.0

    marker.scale.x = 0.05
    marker.scale.y = 0.05
    marker.scale.z = 0.05

    marker.color.r = 1
    marker.color.g = 0
    marker.color.b = 0
    marker.color.a = 1

    marker.lifetime = rospy.Duration(0)


    my_point = Point()

    for filename in files_to_read:

        counter = rospy.ServiceProxy("/give_next_pc", GiveNextPc)
        response = counter(filename)
        rospy.loginfo("ply file number :" + str(response.count))
        plyfile = os.path.join(path_to_files, filename)

        xyz_pcd, bbox_bgr_pcd, b_boxes, b_box_img = create_newpoints(plyfile)

        if len(b_boxes) != 0:


            bbox_full_pcd_list = []
            marker_coord_list = []
            print(xyz_pcd.shape)
            print(bbox_bgr_pcd.shape)
            bbox_full_pcd_vis = np.concatenate((xyz_pcd, bbox_bgr_pcd), axis=1)
            refresh(bbox_full_pcd_vis)

            for i in range(len(b_boxes)):

                trunk_rgb_pcd = np.reshape(b_boxes[i], (407040, 3))
                bbox_full_pcd = np.concatenate((xyz_pcd, bbox_bgr_pcd), axis=1)
                bbox_full_pcd_list.append(bbox_full_pcd)

                marker_coord = get_xyz(bbox_full_pcd_list[i])
                marker_coord_list.append(marker_coord)

            PC_MESSAGE.header.stamp = rospy.Time.now()
            pub.publish(PC_MESSAGE)



            for i in range(len(b_boxes)):



                marker.id = i

                my_point.z = marker_coord_list[i][2]
                my_point.x = marker_coord_list[i][0]
                my_point.y = marker_coord_list[i][1]

                marker.pose.position = my_point

                marker_array.markers.append(marker)

            marker_array_publisher.publish(marker_array)



            for i in range(len(b_boxes)):
                marker.id = i
                marker.ns = ''
                marker.action = Marker.DELETEALL
                marker_array.markers.append(marker)
            marker_array_publisher.publish(marker_array)


if __name__ == "__main__":

    visualize_target()
