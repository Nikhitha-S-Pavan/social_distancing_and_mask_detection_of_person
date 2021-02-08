from configs.detector import detect_people
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import configs


# Reading config, weights and labels
net = cv2.dnn.readNet('configs/yolov3_mask_person_last.weights', 'configs/yolov3_mask_person.cfg')
classes = []


LABELS = open('configs/classes.txt').read().strip().split("\n")

print(LABELS)

# determine only the "output" layer names that we need from YOLO

ln = net.getUnconnectedOutLayersNames()
print(ln)

vs = cv2.VideoCapture("vid2.mp4")

while True:
    # read the next frame from the input video
    (grabbed, frame) = vs.read()
    # if the frame was not grabbed, then that's the end fo the stream
    if not grabbed:
        break
    # resize the frame and then detect people (only people) in it
    frame = imutils.resize(frame, width=700)

    results = detect_people(frame, net, ln, personIdx=LABELS.index("person"), maskIdx=None, nomaskIdx=None)
    mask_nomask = detect_people(frame, net, ln, personIdx=None, maskIdx=LABELS.index("mask"), nomaskIdx=LABELS.index("nomask"))
    #mask_list = [LABELS[0], LABELS[2]]
    #results = detect_people(frame, net, personIdx=mask_list)
    # initialize the set of indexes that violate the minimum social distance
    violate = set()

    # ensure there are at least two people detections (required in order to compute the
    # the pairwise distance maps)
    if len(results) >= 2:
        # extract all centroids from the results and compute the Euclidean distances
        # between all pairs of the centroids
        centroids = np.array([r[2] for r in results])
        D = dist.cdist(centroids, centroids, metric="euclidean")

        # loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
                # check to see if the distance between any two centroid pairs is less
                # than the configured number of pixels
                if D[i, j] < 200:
                    # update the violation set with the indexes of the centroid pairs
                    violate.add(i)
                    violate.add(j)

    # loop over the results
    for (i, (prob, bbox, centroid)) in enumerate(results):
        # extract teh bounding box and centroid coordinates, then initialize the color of the annotation
        (startX, startY, endX, endY) = bbox
        (cX, cY) = centroid
        color = (0, 255, 0)

        # if the index pair exists within the violation set, then update the color
        if i in violate:
            color = (0, 0, 255)

        # draw (1) a bounding box around the person and (2) the centroid coordinates of the person
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        cv2.circle(frame, (cX, cY), 5, color, 1)

    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(violate))
    cv2.putText(frame, text, (10, frame.shape[0] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    # check to see if the output frame should be displayed to the screen

    # show the output frame
    cv2.imshow("Output", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

    # provide video path to save
    # initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter("C:\\Users\\NIU2KOR\\Desktop\\Out.mp4", fourcc, 25, (frame.shape[1], frame.shape[0]), True)

    # if the video writer is not None, write the frame to the output video file
    if writer is not None:
        print("[INFO] writing stream to output")
        writer.write(frame)




