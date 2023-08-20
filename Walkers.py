import cv2
import numpy as np

# Create our body classifier
body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')

# Initiate video capture for video file
thing = cv2.VideoCapture('walking.avi')

# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = thing.read()
    #frame = cv2.resize(frame, None,fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies = body_classifier.detectMultiScale(gray, 1.2, 3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        cv2.imshow('Pedestrians', frame)

    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

thing.release()
cv2.destroyAllWindows()