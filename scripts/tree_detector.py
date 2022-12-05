#!/usr/bin/env python


import cv2
import numpy as np



net = cv2.dnn.readNet('/home/woo/catkin_ws/src/orchard_mapping/weights/yolov3_training_1800.weights', '/home/woo/catkin_ws/src/orchard_mapping/config/yolov3_testing.cfg')
classes = ['trunk']

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))


def detect_trees(img):
    """Detect apple trees from an image
    @param img is a BGR image of trees in an orchard that were taken while moving in the orchard
    @returns an image with bounding box and the boxes'edges' coordinates
    """

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
    # print("indexes_flatten: ", indexes.flatten())
    # print("boxes: ", boxes)
    # print("x,y,w,h :", boxes[1])
    # print("class_ids: ", class_ids)
    # print("colors: ", colors)
    # print("confidences: ", confidences)
    # colors = colors[0]

    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) != 0:
        for i in indexes.flatten():
            # print("i:", i)
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color=color, thickness=2)
            final_boxes.append([x,y,w,h])
            final_confidences.append(confidence)

    for finbox in final_boxes:
        finbox[2] = finbox[0] + finbox[2]
        finbox[3] = finbox[1] + finbox[3]

        if finbox[0] < 0:
            finbox[0] = 0
        if finbox[1] < 0:
            finbox[1] = 0


        if finbox[2] >= width - 2:
            finbox[2] = width - 2
        if finbox[3] >= height - 2:
            finbox[3] = height - 2


    final_boxes.sort(key=lambda x:x[0])
    return img, final_boxes


