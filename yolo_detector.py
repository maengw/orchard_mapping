#!/usr/bin/env python3


import cv2
import numpy as np



net = cv2.dnn.readNet('/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_training_1800.weights', '/home/woo/PycharmProjects/maeng_yolo_trunk/yolov3_testing.cfg')
classes = ['trunk']

layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))


def detect_boxes(img):
    """
    img: BGR image
    returns an image with bounding box and the boxes' coordinates
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

    #font = cv2.FONT_HERSHEY_PLAIN
    #colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) != 0:
        #print(indexes)
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color=color, thickness=2)
            # cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 255, 255), 2)
            #cv2.putText(img, label + " " + confidence, (x, 235), font, 2, color, 2)
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


def convert_scale(img, target_type_min, target_type_max, target_type):
    """
    convert data type and rescale the data
    img: bgr image
    target_type_min: target minimum value
    target_type_max: target maximum value
    target_type: target data type
    returns a new image with changed data type with new scale
    """

    imin = 0
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img


