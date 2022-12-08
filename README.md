# orchard_mapping
Code to map an apple orchard by rows with the data collected by Intel Realsense D435 RGBD camera and Trimble RTK-GPS installed on a Gator vehicle driven by human.
Yolo V3 CNN was used to detect trees from RGB images and processed pointcloud to filter background trees and estimate position of the detected tree. Then transformation on estimated tree pose in camera frame with GPS data in world frame were applied in order to obtatin the tree's position in world frame.   

   
# Requirements   
Test Was done on Ubuntu 18.04 OS, python 3.6 and used ROS Melodic but theoretically it will work on any python3++.   
 Ubuntu 18.04   
 Python 3.6   
 ROS1 Melodic   
   

# Installation
1. Clone this repo   
2. Place .plyfiles collected in the orchard in /orchard_mapping/data/sample_plydata   
3. Place .csv file of vehicle's gps x and y data collected in the orchard in /orchard_mapping/data   
4. Check files directories in scripts and edit as instructed in comments in the script files   

   
# Instructions
## Tree detection 
YOLO V3 Convolutional Neural Network was trained on tree images collected in various weather conditions(different time of the year and day). It detects trees in an image and gives edge pixel coordinates of the bounding box.   
* Download yolov3_training_1800.weights file from https://drive.google.com/file/d/1AxkdHrjRh8lwUlIIfGQtGxklDFjq2Vot/view?usp=share_link and place it in /weights/ directory
* /scripts/tree_detector.py : yolov3 network that detects trees   
* /config/yolov3_testing.cfg : network configuration file   
* /weights/yolov3_training_1800.weights : weights file from training network on approximately 1000 tree images for 1800 iterations   

## Pointcloud processing
Process or read .ply file so that it can used for individual's purpose(e.g. visualization, filtering, etc).   
* /scripts/pc_utils.py : used an opensource code; utility functions for reading .ply files and output an array of xyzrgb of rgbd image taken   

## Tree position estimation in camera frame
Reconstruted a RGB image and XYZ 3d array whose shape is the same as RGB image from .ply file and fed RGB image to the detector network for tree detection. Once the network outputs the coordinates of bounding box, points filtering based on the distance was peroformed. Points whose distance(z) are smaller than 0.3m and larger than 1.5m are removed as the objects that are too close(probably leaves) and the objects that are too far(most likely detection of background tree) are unwanted. After filtering unwanted points by distance, then I filtered the points by height(y) to remove the ground points. Tree pose estimation in camera frame was done after removing unwanted points by calculating average of remaining points of each channel(x, y, z).   
* /scripts/depth_utils.py : utility functions for filtering pixels by depth and estimating tree position   

## Vehicle heading and tree frame transformation
Vehicle was driven by human with RTK-GPS installed on it. Position of the vehicle was obtained from GPS and the heading of it was calculated from arctan(previous vehicle position and current vehicle position). Heading of the vehicle is used for tree's frame transformation(camera frame -> world frame).   
* /scripts/map_utils.py : utility functions for tree localization in world frame   

## Tree instance tracking
Object tracking(Tree tracking) is mandatory to prevent counting same tree multiple times in consecutive image frames. There are two conditions that need to be satisfied in order to be classifed as a same tree in frames. Condition1 setting a threshold for horizontal distance between the tree's center in pixel coordinates. As the camera continously collected RGBD data while vehicle was driven, the distance between the tree in frame 1 and frame 2 should be within certain threshold. Condition2 is setting threshold for actual distance between trees in world frame. Since the orchard is a semi-structured environment(distance between different trees are set within certain range), two trees are different trees if the distance between them are bigger than the threshold. A unique tree id was given to each tree and saved all the estimated position of it to use it for mapping. This method was not perfect and it sometimes failed tracking the instances. Hence, I tried using different method to improve tracking.   
A vehicle was only driven in forward direction when collecting data, I changed condition1 to be wether the tree in frame1 is on the right compared to the tree in frame2. Changing condition1 improved the performance of tracking, however, more effort to improve tracking is needed.   
* /scripts/generate_map.py : A main file to generate a map and has instance tracker in the script   

## Find pixel threshold for instance tracking
Distance between vehicle's previous position and current position can be used to guess how much the tree moved from previous frame to current frame in pixel coordinates. If the distance vehicle travelled is far(or short) then the distance between tree in frame1 and frame2 would be far(or short) as well. I generated a new data which has vehicle travel distance and tree pixel displacement and implemented linear regression to obtain the relation between them(basically a linear equation; input: vehicle travel distance; output: expected tree horizontal displacement). As the distance travelled is known from GPS data, displacement of tree can be predicted. I multiplied by 1.3 and used it as threshold for condition 1 in Tree instance tracking. 
* scripts/convert_gps_to_img_state.py   
* /data/tree_gps_img_relation_dist2.csv   

## Generating a map
All trees have unique id and they may have multiple estimated positions as I saved them all in dictionary. A final coordinate of each tree is calculated by averaging the estimated positions of the t
ree. Using just one estimated position as final position of tree is risky because it would be off if there was an error in the steps below. Thus, averaging all potential positions of tree can prevent this and will leverage even if there is an estimated position with huge error.   
* /scripts/generate_map.py : Generate a map of an orchard and save tree's coordinates in .csv file   
   
# Run
## Visualize filtered Pointcloud and target tree's est. position
Pointcloud visualization is done using ROS rviz. New PointCloud2 message that went through processing and filtering is generated and published to a topic and visualized it in rviz. In order to confirm that the tree position estimation is working, Marker message with estimated (x,y,z) is generated and publushed as well and was visualized in rviz. The marker was on the target tree as expected.   
* $ Roslaunch orchard_mapping pc_visualizer.launch   

## Generate a map of an orchard and create .csv file with tree's coordinates
Open two terminals and in first terminal run,   
* $ roscore   
and in second terminal, cd into /orchard_mapping/scripts/ then run,   
* $ python generate_map.py   

