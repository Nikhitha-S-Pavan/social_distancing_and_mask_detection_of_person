import cv2
import numpy as np


def detect_people(frame, net, ln, personIdx=None, maskIdx=None, nomaskIdx=None):

   # grab dimensions of the frame and initialize the list of results
   (H, W) = frame.shape[:2]
   results = []
   # construct a blob from the input frame and then perfrom a forward pass
   # of the YOLO object detector, giving us the bounding boxes and
   # associated probabilities
   blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
   net.setInput(blob)
   layer_outputs = net.forward(ln)

   # initialize lists of detected bounding boxes, centroids, and confidence
   # define list to store data
   boxes=[]
   confidences=[]
   class_ids = []
   centroids = []
   classes = ["mask", "person", "nomask"]
   for layer in layer_outputs:
      for detection in layer:
         scores = detection[5:]
         class_id = np.argmax(scores)
         confidence = scores[class_id]
         if class_id == personIdx and confidence > 0.4:
            center_x = int(detection[0]*W)
            center_y = int(detection[1]*H)
            w = int(detection[2]*W)
            h = int(detection[3]*(H))
            x = int(center_x-w/2)
            y = int(center_y-h/2)
            boxes.append([x,y,w,h])
            class_ids.append(class_id)
            confidences.append(float(confidence))
            centroids.append((center_x, center_y))
         if (class_id == maskIdx and confidence > 0.4) or (class_id == nomaskIdx and confidence > 0.4) :
            center_x = int(detection[0]*W)
            center_y = int(detection[1]*H)
            w = int(detection[2]*W)
            h = int(detection[3]*(H))
            x = int(center_x-w/2)
            y = int(center_y-h/2)
            boxes.append([x,y,w,h])
            class_ids.append(class_id)
            confidences.append(float(confidence))
            centroids.append((center_x, center_y))


   indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.5,0.4)
   font = cv2.FONT_HERSHEY_PLAIN

   if len(indexes) > 0:
      for i in indexes.flatten():
         x,y,w,h=boxes[i]
         label = classes[class_ids[i]]
         confidence = str(round(confidences[i], 2))
         if personIdx:
            # update the results list to consist of the person prediction probability,
            # bounding box coordinates, and the centroid
            r = (confidences[i], (x, y, x + w, y + h), centroids[i])
            results.append(r)
         if label == "mask":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.putText(frame, label + " " + confidence, (x, y + 70), font, 1, (255, 0, 0), 1)
            results = None
         if label == "nomask":
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,0,255), 1)
            print(label)
            cv2.putText(frame, label + " " + confidence, (x, y + 70), font, 1, (0,0,255), 1)
            results = None

   # return the list of results
   return results

