import cv2
import sys

video_capture = cv2.VideoCapture(0)		#<VideoCapture object>

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  #convert color space BGR to GRAY

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):

        break
