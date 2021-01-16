"""
detect_video.py:  A simple python application demonstrating Text Detection in videos using opencv.
"""

__author__ = "S Sathish Babu"
__date__   = "16-01-2021 Saturday 10:00"
__email__  = "bumblebee211196@gmail.com"

import argparse

import cv2
import numpy as np

from imutils.object_detection import non_max_suppression

parser = argparse.ArgumentParser('TextDetection')
parser.add_argument('-v', '--video', help='Path to the video file', default=0)
parser.add_argument('-c', '--confidence', help='Confidence score for threshold', default=0.5, type=float)
args = vars(parser.parse_args())

net = cv2.dnn.readNet('./resources/frozen_east_text_detection.pb')
# If GPU is available
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = [
    'feature_fusion/Conv_7/Sigmoid',
    'feature_fusion/concat_3'
]

vc = cv2.VideoCapture(args['video'])
confidence = args['confidence']
writer = None

while True:
    _, image = vc.read()
    original = image.copy()
    h, w = image.shape[:2]
    new_h, new_w = 320, 320
    rh, rw = h / float(new_h), w / float(new_w)
    image = cv2.resize(image, (new_w, new_h))
    h, w = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores, geometry = net.forward(layer_names)

    rows, cols = scores.shape[2:4]
    rects, confidences = [], []

    for y in range(0, rows):
        score_data = scores[0, 0, y]
        x_data0 = geometry[0, 0, y]
        x_data1 = geometry[0, 1, y]
        x_data2 = geometry[0, 2, y]
        x_data3 = geometry[0, 3, y]
        angle_data = geometry[0, 4, y]

        for x in range(0, cols):
            if score_data[x] > confidence:
                offset_x, offset_y = x * 4.0, y * 4.0
                angle = angle_data[x]
                cos, sin = np.cos(angle), np.sin(angle)
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                x2 = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                y2 = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                x1 = int(x2 - w)
                y1 = int(y2 - h)
                rects.append((x1, y1, x2, y2))
                confidences.append(score_data[x])

    boxes = non_max_suppression(np.array(rects), probs=confidences)

    for x1, y1, x2, y2 in boxes:
        x1 = int(x1 * rw)
        y1 = int(y1 * rh)
        x2 = int(x2 * rw)
        y2 = int(y2 * rh)
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('Output', original)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()
