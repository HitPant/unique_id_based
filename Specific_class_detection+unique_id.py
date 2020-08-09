from imutils.video import VideoStream
from tracker.centroidtracker import CentroidTracker
from imutils.video import FPS
import numpy as np
import imutils
import time
import cv2



# Mobile net SSD Model from Caffe
prtxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'
# Confidence threshold
conf = .75

ct = CentroidTracker()
(H, W) = (None, None)

# initialize the list of class labels MobileNet SSD was trained to
# detect with the caffe model.
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

# Load the Model
net = cv2.dnn.readNetFromCaffe(prtxt, model)

# Randomize some colors for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))



# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=800)

    # grab the frame dimensions and convert it to a blob
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (W, H)),
                                 0.007843, (300, 300), 127.5)

    # pass the blob through the network and obtain the detections and
    # predictions
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # list for persons, dont want person box inside another
    personsFound = []
    confHigh = 0

    # WHether to draw box or not
    drawBox = True

    # loop over the detections
    for i in np.arange(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        # if confidence > conf:
        #
        #     # extract the index of the class label from the
        #     # `detections`, then compute the (x, y)-coordinates of
        #     # the bounding box for the object
        idx = int(detections[0, 0, i, 1])
        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
        rects.append(box.astype("int"))

            # persons checker
        if CLASSES[idx] == 'person':
            # if i == 0:
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)


    objects = ct.update(rects)
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

    # update the FPS counter
    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
