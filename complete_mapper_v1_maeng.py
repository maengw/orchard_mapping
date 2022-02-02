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
# from localisation.srv import LoadOctomap
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

net = cv2.dnn.readNet('/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_training_1800.weights', '/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_testing.cfg')
classes = ['trunk']



# All ply reading utils courtesy of:
# https://github.com/daavoo/pyntcloud/blob/master/pyntcloud/io/ply.py

sys_byteorder = ('>', '<')[sys.byteorder == 'little']

ply_dtypes = dict([
    (b'int8', 'i1'),
    (b'char', 'i1'),
    (b'uint8', 'u1'),
    (b'uchar', 'b1'),
    (b'uchar', 'u1'),
    (b'int16', 'i2'),
    (b'short', 'i2'),
    (b'uint16', 'u2'),
    (b'ushort', 'u2'),
    (b'int32', 'i4'),
    (b'int', 'i4'),
    (b'uint32', 'u4'),
    (b'uint', 'u4'),
    (b'float32', 'f4'),
    (b'float', 'f4'),
    (b'float64', 'f8'),
    (b'double', 'f8')
])

valid_formats = {'ascii': '', 'binary_big_endian': '>',
                 'binary_little_endian': '<'}


layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

def create_bbox(img):

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    net.setInput(blob)

    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    final_boxes = []
    final_confidences = []

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


    if len(indexes) != 0:
        #print(indexes)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color=color, thickness=2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)
            cv2.putText(img, label + " " + confidence, (x, 235), font, 2, color, 2)
            final_boxes.append([x,y,w,h])
            final_confidences.append(confidence)


    for finbox in final_boxes:
        finbox[2] = finbox[0] + finbox[2]
        finbox[3] = finbox[1] + finbox[3]

        if finbox[0] < 0:
            finbox[0] = 0
        if finbox[1] < 0:
            finbox[1] = 0

        if finbox[2] > 846:
            finbox[2] = 846
        if finbox[3] > 478:
            finbox[3] = 478

    final_boxes.sort(key=lambda x:x[0])
    return img, final_boxes


def convert(img, target_type_min, target_type_max, target_type):

    imin = 0
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


def read_ply(filename):

    """ Read a .ply (binary or ascii) file and store the elements in pandas DataFrame
    Parameters
    ----------
    filename: str
        Path to the filename
    Returns
    -------
    data: dict
        Elements as pandas DataFrames; comments and ob_info as list of string
    """

    with open(filename, 'rb') as ply:

        if b'ply' not in ply.readline():
            raise ValueError('The file does not start whith the word ply')
        # get binary_little/big or ascii
        fmt = ply.readline().split()[1].decode()
        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        line = []
        dtypes = defaultdict(list)
        count = 2
        points_size = None
        mesh_size = None
        has_texture = False
        comments = []
        while b'end_header' not in line and line != b'':
            line = ply.readline()

            if b'element' in line:
                line = line.split()
                name = line[1].decode()
                size = int(line[2])
                if name == "vertex":
                    points_size = size
                elif name == "face":
                    mesh_size = size

            elif b'property' in line:
                line = line.split()
                # element mesh
                if b'list' in line:

                    if b"vertex_indices" in line[-1] or b"vertex_index" in line[-1]:
                        mesh_names = ["n_points", "v1", "v2", "v3"]
                    else:
                        has_texture = True
                        mesh_names = ["n_coords"] + ["v1_u", "v1_v", "v2_u", "v2_v", "v3_u", "v3_v"]

                    if fmt == "ascii":
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ply_dtypes[line[3]]
                    else:
                        # the first number has different dtype than the list
                        dtypes[name].append(
                            (mesh_names[0], ext + ply_dtypes[line[2]]))
                        # rest of the numbers have the same dtype
                        dt = ext + ply_dtypes[line[3]]

                    for j in range(1, len(mesh_names)):
                        dtypes[name].append((mesh_names[j], dt))
                else:
                    if fmt == "ascii":
                        dtypes[name].append(
                            (line[2].decode(), ply_dtypes[line[1]]))
                    else:
                        dtypes[name].append(
                            (line[2].decode(), ext + ply_dtypes[line[1]]))

            elif b'comment' in line:
                line = line.split(b" ", 1)
                comment = line[1].decode().rstrip()
                comments.append(comment)

            count += 1

        # for bin
        end_header = ply.tell()

    if fmt == 'ascii':
        top = count
        bottom = 0 if mesh_size is None else mesh_size

        names = [x[0] for x in dtypes["vertex"]]

        pts = pd.read_csv(filename, sep=" ", header=None, engine="python",
                                     skiprows=top, skipfooter=bottom, usecols=names, names=names)

        for n, col in enumerate(pts.columns):
            pts[col] = pts[col].astype(
                dtypes["vertex"][n][1])
        pts = pts.values
        if not np.abs(pts[-1]).sum():
            pts = pts[:-1]

        return pts

    else:
        raise NotImplementedError("Non-ASCII PLY file!")


def write_ply(filename, points=None, mesh=None, as_text=False, comments=None):
    """

    Parameters
    ----------
    filename: str
        The created file will be named with this
    points: ndarray
    mesh: ndarray
    as_text: boolean
        Set the write mode of the file. Default: binary
    comments: list of string

    Returns
    -------
    boolean
        True if no problems

    """
    if not filename.endswith('ply'):
        filename += '.ply'

    # open in text mode to write the header
    with open(filename, 'w') as ply:
        header = ['ply']

        if as_text:
            header.append('format ascii 1.0')
        else:
            header.append('format binary_' + sys.byteorder + '_endian 1.0')

        if comments:
            for comment in comments:
                header.append('comment ' + comment)

        if points is not None:
            header.extend(describe_element('vertex', points))
        if mesh is not None:
            mesh = mesh.copy()
            mesh.insert(loc=0, column="n_points", value=3)
            mesh["n_points"] = mesh["n_points"].astype("u1")
            header.extend(describe_element('face', mesh))

        header.append('end_header')

        for line in header:
            ply.write("%s\n" % line)

    if as_text:
        if points is not None:
            points.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                          encoding='ascii')
        if mesh is not None:
            mesh.to_csv(filename, sep=" ", index=False, header=False, mode='a',
                        encoding='ascii')

    else:
        with open(filename, 'ab') as ply:
            if points is not None:
                points.to_records(index=False).tofile(ply)
            if mesh is not None:
                mesh.to_records(index=False).tofile(ply)

    return True


def describe_element(name, df):
    """ Takes the columns of the dataframe and builds a ply-like description

    Parameters
    ----------
    name: str
    df: pandas DataFrame

    Returns
    -------
    element: list[str]
    """
    property_formats = {'f': 'float', 'u': 'uchar', 'i': 'int'}
    element = ['element ' + name + ' ' + str(len(df))]

    if name == 'face':
        element.append("property list uchar int vertex_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append('property ' + f + ' ' + df.columns.values[i])

    return element

def convert_points_to_pointcloud(points, frame_id):

    # points is a Python list with [x, y, z, (r, g, b)]  - Colors will be detected based on first element length
    # can also be a numpy array
    # colors is either None if you don't want RGB, or an Nx3 numpy array with the corresponding RGB values as integers from 0 to 255

    # Heavily sourced from: https://gist.github.com/lucasw/ea04dcd65bc944daea07612314d114bb
    # And: https://answers.ros.org/question/289576/understanding-the-bytes-in-a-pcl2-message/

    dim = len(points[0])
    if dim == 3:
        colors = False
    elif dim == 6:
        colors = True
    else:
        raise ValueError('Unsure how to interpret dimension {} input'.format(dim))

    if isinstance(points, np.ndarray):
        points = points.tolist()


    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    fields = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
    ]
    if colors:
        fields.append(PointField('rgb', 12, PointField.UINT32, 1))
        formatted_points = []
        for row in points:
            xyz = row[:3]
            r, g, b = row[3:]
            r = np.uint8(r)
            g = np.uint8(g)
            b = np.uint8(b)


            a = 255
            rgba = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
            xyz.append(rgba)
            formatted_points.append(xyz)

        points = formatted_points

    pc = pc2.create_cloud(header, fields, points)
    return pc


def refresh(filetoread):

    pc = read_ply(filetoread)
    n = len(pc)


    pc_msg = convert_points_to_pointcloud(pc, frame_id='map')


    global PC_MESSAGE
    PC_MESSAGE = pc_msg

    return pc

PC_MESSAGE = None



def do_transformation(cam_tree_x, cam_tree_y, gps_x, gps_y, vehicle_heading, camera_heading):

    T_wv = np.array([[np.cos(vehicle_heading), -np.sin(vehicle_heading), gps_x], [np.sin(vehicle_heading), np.cos(vehicle_heading), gps_y], [0, 0, 1]])

                                                           
    T_vc = np.array([[np.cos(camera_heading), -np.sin(camera_heading), 0.7], [np.sin(camera_heading), np.cos(camera_heading), 0], [0, 0, 1]])

    tree_c = np.array([[cam_tree_x], [cam_tree_y], [1]])

    tree_w_xy = np.dot(np.dot(T_wv, T_vc), tree_c)


    return tree_w_xy[0][0], tree_w_xy[1][0]


def get_distance(alltx1, allty1, alltx2, allty2):

    dist = np.sqrt( (alltx2 - alltx1)**2 + (allty2 - allty1)**2 )
    return dist    


def draw_map(alltx, allty):

    dists = []
    new_instance_indexes = []

    for i in range(len(alltx)-1):
        dists.append(get_distance(alltx[i], allty[i], alltx[i+1], allty[i+1]))

    for j in range(len(dists)):

        if dists[j] > 0.955:
            new_instance_indexes.append(j)

    tx = []
    ty = []


    k = 0
    newidxlen = len(new_instance_indexes)
    print(newidxlen)
    for idx in new_instance_indexes:
        if k == 0:
            calx = alltx[:idx+1]
            caly = allty[:idx+1]
            avgx = sum(calx) / len(calx)
            avgy = sum(caly) / len(caly)

        elif k < newidxlen-1:
            calx = alltx[idx+1 : new_instance_indexes[k+1]+1]
            caly = allty[idx+1 : new_instance_indexes[k+1]+1]
            avgx = sum(calx) / len(calx)
            avgy = sum(caly) / len(caly)

        tx.append(avgx)
        ty.append(avgy)
        k += 1

    return tx, ty


def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]

    else:
        dictionary[key].append(value)


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

        print("image number: ", t)


        # code to calculate heading
        if t == 0:
            vehicle_heading = np.arctan2(vehicle_gps_y_list[0] - 2*(vehicle_gps_y_list[1] - vehicle_gps_y_list[0]), vehicle_gps_x_list[0] - 2*(vehicle_gps_x_list[1] - vehicle_gps_x_list[0]))
        else:
            vehicle_heading = np.arctan2(vehicle_gps_y_list[t] - vehicle_gps_y_list[t-1], vehicle_gps_x_list[t] - vehicle_gps_x_list[t-1])

        b = os.path.join(path_to_files, a)
        pcd = refresh(b)

        pub = rospy.Publisher(publish_topic, PointCloud2, queue_size=10,latch=True)
        rate = rospy.Rate(1)

        PC_MESSAGE.header.stamp = rospy.Time.now()
        pub.publish(PC_MESSAGE)
    
        
        ori_xyz_pcd = pcd[:, 0:3]
        ori_rgb_pcd = pcd[:, 3:6]
        ori_rgb_img = np.reshape(ori_rgb_pcd, (480, 848, 3)) / 255
        ori_rgb_img = convert(ori_rgb_img, 0, 255, np.uint8)
        ori_bgr_img = cv2.cvtColor(ori_rgb_img, cv2.COLOR_RGB2BGR)


        # bbox_bgr_img is an image with bbox on the detected trunks and boxes is list of lists with [x,y,w,h].
        bbox_bgr_img, boxes = create_bbox(ori_bgr_img)


        ori_xyz_pcd = np.reshape(ori_xyz_pcd, (480, 848, 3))

        bbox_points_list = []


        if len(boxes) != 0:

            for box in boxes:

                x1 = box[0]
                x2 = box[2]
                y1 = box[1]
                y2 = box[3]

                abox = ori_xyz_pcd[y1:y2, x1:x2, :]
                bbox_points_list.append(abox)



            valid_bbox_points_x_array = []
            valid_bbox_points_y_array = []
            valid_bbox_points_z_array = []


            waha = 0

            centers = []
            for bbox_point in bbox_points_list:
                bbox_point_x = bbox_point[:, :, 0]
                bbox_point_y = bbox_point[:, :, 1]
                bbox_point_z = bbox_point[:, :, 2]    

                n = bbox_point.shape[0]
                nn = bbox_point.shape[1]


                if n != 0 and nn != 0:                
                    valid_bbox_points_x_list = []
                    valid_bbox_points_y_list = []
                    valid_bbox_points_z_list = []   


                    num_of_z_points = len(bbox_point_z.flatten())

                    for i in range(n):

                        for j in range(nn):
                            if np.isnan(bbox_point_x[i][j]) or np.isnan(bbox_point_y[i][j]) or np.isnan(bbox_point_z[i][j]):
                                pass

                            elif bbox_point_z[i][j] <= 1.5:
                                valid_bbox_points_z_list.append(bbox_point_z[i][j])
                                valid_bbox_points_x_list.append(bbox_point_x[i][j])
                                valid_bbox_points_y_list.append(bbox_point_y[i][j])

                    num_of_filtered_z_points = len(valid_bbox_points_z_list)

                    foreground_ratio = num_of_filtered_z_points / num_of_z_points

                    if foreground_ratio >= 0.188:


                        remover_index = []

                        for k in range(len(valid_bbox_points_y_list)):
                            if valid_bbox_points_y_list[k] > 0.3:
                                remover_index.append(k)

                        remover_index.sort(reverse=True)

                        for index in remover_index:
                            valid_bbox_points_z_list.pop(index)
                            valid_bbox_points_y_list.pop(index)
                            valid_bbox_points_x_list.pop(index)   

                        x1 = boxes[waha][0]
                        x2 = boxes[waha][2]
                        y1 = boxes[waha][1]
                        y2 = boxes[waha][3]


                        centers.append((x1+x2)/2)                    

                        valid_bbox_points_x_array.append(valid_bbox_points_x_list) 
                        valid_bbox_points_y_array.append(valid_bbox_points_y_list) 
                        valid_bbox_points_z_array.append(valid_bbox_points_z_list)

                    else:
                        pass

                waha += 1


            valid_bbox_points_y_nparray = np.array(valid_bbox_points_y_array)


            if len(valid_bbox_points_y_nparray) != 0:

                hawa = 0
                txw = []
                tyw = []

                for l in range(len(valid_bbox_points_y_nparray)):
    
                    argsorted_y_nparray = np.argsort(valid_bbox_points_y_nparray[l])[::-1]
                    valid_indexes = argsorted_y_nparray[:int(len(valid_bbox_points_y_nparray[l])/10)]

                    x_coords = []
                    y_coords = []
                    z_coords = []

                    for indice in valid_indexes:
                        x_coords.append(valid_bbox_points_x_array[l][indice])
                        y_coords.append(valid_bbox_points_y_array[l][indice])
                        z_coords.append(valid_bbox_points_z_array[l][indice])


                    if len(x_coords) != 0 and len(y_coords) != 0 and len(z_coords) != 0:

                        target_x = sum(x_coords) / len(x_coords)
                        target_y = sum(y_coords) / len(y_coords)
                        target_z = sum(z_coords) / len(z_coords)


####                    # Calculate distance to the tree
                        dist_to_tree = np.sqrt(target_x**2 + target_z**2)
                                                                                                                                       # - np.pi/2
                        tx_w, ty_w = do_transformation(target_x, target_z, vehicle_gps_x_list[t], vehicle_gps_y_list[t], vehicle_heading           , 0)# camera_heading)


                        txw.append(tx_w)
                        tyw.append(ty_w)



                    else:
                        pass
                        # print("coordinates are not saved since the length of x or y z coordinates are 0")


                print("Valid image number: ", tt)
                if t == 0 : # First initiation
                    if len(centers) > 0:
                        if len(centers) == 1:
                            set_key(image_dict, key=tt, value=[centers[0], txw[0], tyw[0]])
                        else: # when there's more than 1 tree in the first image. This could be ignored, since the beginning will only have 1 tree.
                            for i in range(len(centers)):
                                set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                set_key(tree_dict, key=tree_id, value=[txw[i], tyw[i]])
                                tree_id += 1

                else:
                    if len(centers) > 0:
                        prev_data = image_dict.get(tt-1)
                        #print(prev_data)
                        lengths = len(prev_data)

                       # t=1; a
                        if lengths == 1:
                            if len(centers) == 1:
                                set_key(image_dict, key=tt, value=[centers[0], txw[0], tyw[0]])
                                dist = get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0])
                               # t=2; a

                                if dist < thresh and prev_data[0][0] > centers[0]:    
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                               # t=2; b
                                else:
                                   tree_id += 1
                                   set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])

                            # t=2; (a,b) or (b,c)
                            elif len(centers) == 2:
                                distab = []
                                distbc = []
                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                distab.append(get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0]))
                                #distbc.append(get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0]))
                                #distbc.append(get_distance(prev_data[0][1], prev_data[0][2], txw[1], tyw[1]))
                                # t=2; (a,b)
                                if distab[0] < thresh and prev_data[0][0] > centers[0]:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                # t=2; (b,c)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])

                            elif len(centers) == 3:
                                distabc = []
                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                distabc.append(get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0]))
                                # t=2; (a,b,c)
                                if distabc[0] < thresh and prev_data[0][0] > centers[0]:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (b,c,d)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])

                        # t=1; (a,b)
                        elif lengths == 2:
                            if len(centers) == 1:
                                set_key(image_dict, key=tt, value=[centers[0], txw[0], tyw[0]])
                                dist = get_distance(prev_data[1][1], prev_data[1][2], txw[0], tyw[0])
                                # t=2; (b)
                                if dist < thresh and prev_data[1][0] > centers[0]:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                # t=2; (c)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])

                            # t=2; (a,b) or (b,c) or (c,d)
                            elif len(centers) == 2:
                                distab = []
                                distbc = []

                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                distab.append(get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0]))
                                distab.append(get_distance(prev_data[1][1], prev_data[1][2], txw[1], tyw[1]))
                                distbc.append(get_distance(prev_data[1][1], prev_data[1][2], txw[0], tyw[0]))
                                # t=2; (a,b)
                                if distab[0] < thresh and prev_data[0][0] > centers[0]:  #and dist[1] < thresh and prev_data[1][0]  > centers[1]:
                                    set_key(tree_dict, key=tree_id-1, value=[txw[0], tyw[0]])
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                # t=2; (b,c)
                                elif distbc[0] < thresh and prev_data[1][0] > centers[0]: #and dist[1] > thresh:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                # t=2; (c,d)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])

                            # t=2; (a,b,c) or (b,c,d) or (c,d,e)
                            elif len(centers) == 3:
                                distabc = []
                                distbcd = []
                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                distabc.append(get_distance(prev_data[0][1], prev_data[0][2], txw[0], tyw[0]))
                                distabc.append(get_distance(prev_data[1][1], prev_data[1][2], txw[1], tyw[1]))
                                distbcd.append(get_distance(prev_data[1][1], prev_data[1][2], txw[0], tyw[0]))
                                # t=2; (a,b,c)
                                if distabc[0] < thresh and prev_data[0][0] > centers[0]: #and dist[1] < thresh and prev_data[1][0] > centers[1] and dist[2] > thresh:
                                    set_key(tree_dict, key=tree_id-2, value=[txw[0], tyw[0]])
                                    set_key(tree_dict, key=tree_id-1, value=[txw[1], tyw[1]])
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (b,c,d)
                                elif distbcd[0] < thresh and prev_data[1][0] > centers[0]: #and dist[1] > thresh and dist[2] > thresh:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (c,d,e)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])                                

                        # t=1; (a,b,c)
                        elif lengths == 3:
                            if len(centers) == 1:
                                set_key(image_dict, key=tt, value=[centers[0], txw[0], tyw[0]])
                                dist = get_distance(prev_data[2][1], prev_data[2][2], txw[0], tyw[0])
                                # t=2; (c)
                                if dist < thresh and prev_data[2][0] > centers[0]:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                # t=2; (d)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                            
                            elif len(centers) == 2:
                                distbc = []
                                distcd = []
                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                distbc.append(get_distance(prev_data[1][1], prev_data[1][2], txw[0], tyw[0]))
                                distbc.append(get_distance(prev_data[2][1], prev_data[2][2], txw[1], tyw[1]))
                                distcd.append(get_distance(prev_data[2][1], prev_data[2][2], txw[0], tyw[0]))
                                # t=2; (b,c)
                                if distbc[0] < thresh and prev_data[1][0] > centers[0]: #and distbc[1] < thresh and prev_data[2][0] > centers[1]:
                                    set_key(tree_dict, key=tree_id-1, value=[txw[0], tyw[0]])
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                # t=2; (c,d)
                                elif distcd[0] < thresh and prev_data[2] > centers[0]: #and distbc[1] > thresh and distcd[0] < thresh:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                # t=2; (d,e)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])

                            elif len(centers) == 3:
                                distabc = []
                                distbcd = []
                                distcde = []
                                for i in range(len(centers)):
                                    set_key(image_dict, key=tt, value=[centers[i], txw[i], tyw[i]])
                                    distabc.append(get_distance(prev_data[i][1], prev_data[i][2], txw[i], tyw[i]))
                                distbcd.append(get_distance(prev_data[1][1], prev_data[1][2], txw[0], tyw[0]))
                                distbcd.append(get_distance(prev_data[2][1], prev_data[2][2], txw[1], tyw[1]))
                                #distbcd.append(get_distance(prev_data[2][1], prev_data[2][2], txw[2], tyw[2]))
                                distcde.append(get_distance(prev_data[2][1], prev_data[2][2], txw[0], tyw[0]))
                                #distcde.append(get_distance(prev_data[2][1], prev_data[2][2], txw[1], tyw[1]))

                                # t=2; (a,b,c)
                                if distabc[0] < thresh and prev_data[0] > centers[0]: #and distabc[1] < thresh and prev_data[1] > centers[1] and distabc[2] < thresh and prev_data[2] > centers[2]:
                                    set_key(tree_dict, key=tree_id-2, value=[txw[0], tyw[0]])
                                    set_key(tree_dict, key=tree_id-1, value=[txw[1], tyw[1]])
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (b,c,d)
                                elif distbcd[0] < thresh and prev_data[1] > centers[0]: #and distbcd[1] < thresh and prev_data[2] > centers[1] and distbcd[2] > thresh:
                                    set_key(tree_dict, key=tree_id-1, value=[txw[0], tyw[0]])
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (c,d,e)
                                elif distcde[0] < thresh and prev_data[2] > centers[0]: #and distcde[1] > thresh:
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                                # t=2; (d,e,f)
                                else:
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[0], tyw[0]])
                                    tree_id += 1
                                    set_key(tree_dict, key=tree_id, value=[txw[1], tyw[1]])
                                    tree_id += 1                            
                                    set_key(tree_dict, key=tree_id, value=[txw[2], tyw[2]])
                print("tree id : ", tree_id)
                tt += 1


        t+=1


    if ender == 0:
        rospy.signal_shutdown("Read all the ply files!")

    treesxandy = []


    print(tree_dict)
    for key, val in tree_dict.items():

        treesxandy.append(val)


    valix = []
    valiy = []

    for itm in treesxandy:
        xx = 0
        yy = 0
        for hg in range(len(itm)):
            xx = xx + itm[hg][0]
            yy = yy + itm[hg][1]

        valix.append(xx/len(itm))
        valiy.append(yy/len(itm))

    print('valid x')
    print(valix)
    print()
    print('valid y')
    print(valiy)

    plt.scatter(valix, valiy, s=5, marker='o')
    plt.title('Orchard map')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(45+5.245e5,80+5.245e5)
    plt.ylim(50+5.125e6, 130+5.125e6)
    plt.savefig('/home/woo/Desktop/test_map/final_maps/instance_tracked_2mtds_map.png') 

if __name__ == "__main__":
    main()

