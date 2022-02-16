# orchard_mapping
Code to map an apple orchard with the data collected by Intel Realsense D435 RGBD camera and Trimble RTK-GPS.
Yolo V3 CNN was used to detect trees from the data(RGB Image) and filtered the background trees by depth(pointcloud).
GPS data and the Tree's pose with respect to the camera frame were fused along with transformation in order to get the coordinate of the trees in world frame.

## Tree detection
Yolo V3 network was trained on approximately 1000 images of trees in the orchard under various conditions(Different year and different time of the year and day). 
The network shows better performance on Fall data than the summer data as there are more lighting in summer which whites out the pixles.
Network input is a bgr image and outputs the pixel coordinates of bounding boxes of the trees.

## Background tree filtering and tree pose estimation
Pointcloud collected from Realsense camera gives heiht, width, and depth of each pixel in the image with respect to camera frame.
Mapping process is done by row by row, thus the trees in the background needed to be filtered. Points whose depth are larger(farther) than 1.5m are considered as 
background trees and they are removed.
Tree pose eistimation is done by 3 steps which are,
1. Find the lowest 10% of the points in height
2. Find the corresponding depth and widh points
3. Average the points in depth and width individually
The output from the step 3 is the y and x of the tree with respect to the camera frame.

## Tree instances tracking
Current method of tracking instances of trees for double counting prevention is not accurate. It checks the trees' position both in pixels and in world frame
in consecutive images and then compare the location of the tree and distance between them. This method works only with the assumption of vehicle moving forward at all times.

## Mapping process
Images and the GPS data were collected simultaneously, thus the tree position with respect to the camera frame could be transformed into the world frame with the GPS data 
collected when the image was taken.
