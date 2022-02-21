#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utm

# Transformation matrix for vehicle frame to world frame.
def do_transformation(cam_tree_x, cam_tree_y, gps_x, gps_y, vehicle_heading, camera_heading, dist_btw_cam_gps):
    """
    cam_tree_x: Tree's x position with respect to the camera frame
    cam_tree_y: Tree's y position with respect to the camera frame
    gps_x: Vehicle's gps x in world frame
    gps_y: Vehicle's gps y in world frame
    vehicle_heading: Heading of the vehicle in world frame
    camera_heading: Heading of the camera with respect to the vehicle.
    dist_btw_cam_gps: Distance between the camera and RTK-GPS unit.
    returns tree's x and y in world frame.
    """  

    T_wv = np.array([[np.cos(vehicle_heading), -np.sin(vehicle_heading), gps_x], [np.sin(vehicle_heading), np.cos(vehicle_heading), gps_y], [0, 0, 1]])

                                                           
    T_vc = np.array([[np.cos(camera_heading), -np.sin(camera_heading), dist_btw_cam_gps], [np.sin(camera_heading), np.cos(camera_heading), 0], [0, 0, 1]])

    tree_c = np.array([[cam_tree_x], [cam_tree_y], [1]])

    tree_w_xy = np.dot(np.dot(T_wv, T_vc), tree_c)


    # return tree_w_xy[0][0], tree_w_xy[1][0]
    return tree_w_xy[0,0], tree_w_xy[1,0]

# Calculate distance between two points in GPS.
def get_distance(alltx1, allty1, alltx2, allty2):
    """
    alltx1: Tree #1 x gps in world frame
    allty1: Tree #1 y gps in world frame
    alltx2: Tree #2 x gps in world frame
    allty2: Tree #2 y gps in world frame
    returns distance between two trees in world frame
    """

    # dist = np.sqrt( (alltx2 - alltx1)**2 + (allty2 - allty1)**2 )
    return np.linalg.norm(alltx2 - alltx1)
    # return dist    


# Keys of the dictionary are tree id and the values are the corresponding potential coordinates.
def set_key(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]

    else:
        dictionary[key].append(value)


# Calculate heading of the vehicle.
def get_heading(t, vehiclegpsy, vehiclegpsx):
    """
    t: Variable to check whether the current gps data are first data or if there were previous gps collected
    vehiclegpsy: Vehicle gps y in world frame
    vehiclegpsx: Vehicle gps x in world frame
    returns heading of the vehicle
    """
    if t == 0:
        vehicle_heading = np.arctan2(vehiclegpsy[0] - 2*(vehiclegpsy[1] - vehiclegpsy[0]), vehiclegpsx[0] - 2*(vehiclegpsx[1] - vehiclegpsx[0]))
    else:
        vehicle_heading = np.arctan2(vehiclegpsy[t] - vehiclegpsy[t-1], vehiclegpsx[t] - vehiclegpsx[t-1])

    return vehicle_heading


# Create list of lists of bounding boxes.  
def create_box_list(boxes, ori_xyz_pcd):
    """
    boxes: Bounding boxes from the yolo detector
    ori_xyz_pcd: 3D array that x,y,z (depth)
    returns a list of boxes
    """
    bbox_points_list = []
    for box in boxes:

        x1 = box[0]
        x2 = box[2]
        y1 = box[1]
        y2 = box[3]
        abox = ori_xyz_pcd[y1:y2, x1:x2, :]
        bbox_points_list.append(abox)

    return bbox_points_list


# Filter points that are farther than 1.5m from the camera. This is for background tree filtering.
def create_near_points_list(bbox_points_list):
    """
    bbox_points_list: list of bounding boxes from an image
    returns list of x,y,z points that are filtered by distance and the foreground ratio which is the ratio of near points to far points
    """

    near_bbox_points_x_list = []
    near_bbox_points_y_list = []
    near_bbox_points_z_list = []
    num_of_z_points = []
    num_of_filtered_z_points = []


    # centers = []
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


            num_of_z_points.append(len(bbox_point_z.flatten()))

            for i in range(n):

                for j in range(nn):
                    if np.isnan(bbox_point_x[i][j]) or np.isnan(bbox_point_y[i][j]) or np.isnan(bbox_point_z[i][j]):
                        pass

                    elif bbox_point_z[i][j] <= 1.5:
                        valid_bbox_points_z_list.append(bbox_point_z[i][j])
                        valid_bbox_points_x_list.append(bbox_point_x[i][j])
                        valid_bbox_points_y_list.append(bbox_point_y[i][j])
 
            num_of_filtered_z_points.append(len(valid_bbox_points_z_list))

        near_bbox_points_x_list.append(valid_bbox_points_x_list)
        near_bbox_points_y_list.append(valid_bbox_points_y_list)
        near_bbox_points_z_list.append(valid_bbox_points_z_list)


    if len(num_of_z_points) == 0 :
        foreground_ratio = []
    else:
        foreground_ratio = np.array(num_of_filtered_z_points) / np.array(num_of_z_points)

    # print(foreground_ratio)
    return near_bbox_points_x_list, near_bbox_points_y_list, near_bbox_points_z_list, foreground_ratio, num_of_z_points, num_of_filtered_z_points


# Remove ground points.
def remove_ground_points(i, boxes, valid_bbox_points_x_list, valid_bbox_points_y_list, valid_bbox_points_z_list):
    """
    i: number of bounding boxes to loop through
    boxes: bounding boxes
    valid_bbox_points_x_list: list of x(width) points that are near
    valid_bbox_points_y_list: list of y(height) points that are near
    valid_bbox_points_z_list: list of z(depth) points that are near
    returns x,y,z points and the center of the remaining points in the bounding box with the ground points removal
    """

    # centers = []
    remover_index = []
    valid_bbox_points_x_array = []
    valid_bbox_points_y_array = []
    valid_bbox_points_z_array = []
    remover_index_list = []
    valid_bbox_point_y_list = valid_bbox_points_y_list[i]
    valid_bbox_point_x_list = valid_bbox_points_x_list[i]
    valid_bbox_point_z_list = valid_bbox_points_z_list[i]



    for k in range(len(valid_bbox_point_y_list)):
        if valid_bbox_point_y_list[k] > 0.3:
            remover_index.append(k)

    remover_index.sort(reverse=True)

    for index in remover_index:
        valid_bbox_point_z_list.pop(index)
        valid_bbox_point_y_list.pop(index)
        valid_bbox_point_x_list.pop(index)   

    x1 = boxes[i][0]
    x2 = boxes[i][2]
    y1 = boxes[i][1]
    y2 = boxes[i][3]


    center = (x1+x2) / 2                    


    return center, valid_bbox_point_x_list, valid_bbox_point_y_list, valid_bbox_point_z_list



# Get the points that could be used for estimating the target coordinate. Lowest 10% of the points in height and corresponding x and y.
def create_coord_candidates(valid_bbox_points_x_array, valid_bbox_points_y_array, valid_bbox_points_z_array,i):
    """
    valid_bbox_points_x_array: list of x points that are filtered by depth and with no ground points
    valid_bbox_points_y_array: list of y points that are filtered by depth and with no ground points
    valid_bbox_points_z_array: list of z points that are filtered by depth and with no ground points
    i: number of bounding boxes
    returns list of x, y, z candidates 
    """

    valid_bbox_points_y_nparray = np.array(valid_bbox_points_y_array)
      
    argsorted_y_nparray = np.argsort(valid_bbox_points_y_nparray[i])[::-1]
    valid_indexes = argsorted_y_nparray[:int(len(valid_bbox_points_y_nparray[i])/10)]

    x_coords = []
    y_coords = []
    z_coords = []

    for indice in valid_indexes:
        x_coords.append(valid_bbox_points_x_array[i][indice])
        y_coords.append(valid_bbox_points_y_array[i][indice])
        z_coords.append(valid_bbox_points_z_array[i][indice])

    return x_coords, y_coords, z_coords


# Average the points and get a target tree's x,y,z coordinates.
def create_target_coords(x_coords, y_coords, z_coords):
    """
    x_coords: list of target tree's x coordinate candidates
    y_coords: list of target tree's y coordinate candidates
    z_coords: list of target tree's z coordinate candidates
    returns target tree's coordinate in vehicle frame
    """

    target_x = sum(x_coords) / len(x_coords)
    target_y = sum(y_coords) / len(y_coords)
    target_z = sum(z_coords) / len(z_coords)

    return target_x, target_y, target_z


# Prevent double counting same trees and keep the coordinates of the trees with same id using dictionary.
def track_instances(t, tree_id, centers, image_dict, tree_dict, tree_dict_key, txw, tyw, i, thresh):
    """
    t: image number
    tree_id: tree id; counted different trees
    centers: list of centers of the trees
    image_dict: dictionary to keep images
    tree_dict: dictionary to keep the instances of trees
    tree_dict_key: the key for tree_dict
    txw: tree x coordinate in world frame
    tyw: tree y coordinate in world frame
    i: number of detected trees in an image
    thresh: distance threshold between two different trees
    returns tree id, image and tree dictionary
    """
    if t == 0 : # First initiation
	    if len(centers) > 0:
	        if len(centers) == 1:
	            set_key(image_dict, key=tree_dict_key, value=[centers[0], txw[0], tyw[0]])
	        else: # when there's more than 1 tree in the first image. This could be ignored, since the beginning will only have 1 tree.
	            for i in range(len(centers)):
	                set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
	                set_key(tree_dict, key=tree_id, value=[txw[i], tyw[i]])
	                tree_id += 1

    else:
	    if len(centers) > 0:
	        prev_data = image_dict.get(tree_dict_key-1)
	        lengths = len(prev_data)

	       # t=1; a
	        if lengths == 1:
	            if len(centers) == 1:
	                set_key(image_dict, key=tree_dict_key, value=[centers[0], txw[0], tyw[0]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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
	                set_key(image_dict, key=tree_dict_key, value=[centers[0], txw[0], tyw[0]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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
	                set_key(image_dict, key=tree_dict_key, value=[centers[0], txw[0], tyw[0]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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
	                    set_key(image_dict, key=tree_dict_key, value=[centers[i], txw[i], tyw[i]])
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



    return tree_id, tree_dict, image_dict
    

def draw_map(tree_dict):
    """
    tree_dict: tree dictionary whose key is the tree id and value is the coordinate of the tree
    saves the tree
    """
    treesxandy = []

    for key, val in tree_dict.items():

        treesxandy.append(val)

    valix = []
    valiy = []

    for itm in treesxandy:
        xx = 0
        yy = 0
        for i in range(len(itm)):
            xx = xx + itm[i][0]
            yy = yy + itm[i][1]

        valix.append(xx/len(itm))
        valiy.append(yy/len(itm))

    plt.scatter(valix, valiy, s=5, marker='o')
    plt.title('Orchard map')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(45+5.245e5,80+5.245e5)
    plt.ylim(50+5.125e6, 130+5.125e6)

    plt.xlim(min(valix), max(valix))
    plt.ylim(min(valiy), max(valiy))
    plt.savefig('/home/woo/Desktop/test_map/final_maps/instance_tracked_2mtds_map.png') 


def create_gps_list(csv_file):
    """
    csv_file: vehicle gps csv file with the whole path to directory
    returns gps x and y of the vehicle
    """
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

    return vehicle_gps_x_list, vehicle_gps_y_list

