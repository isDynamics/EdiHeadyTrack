import EdiHeadyTrack as eht
import cv2

# Load the cascade
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')
side_face_cascade = cv2.CascadeClassifier('resources/haarcascade_profileface.xml')
upper_body_cascade = cv2.CascadeClassifier('resources/haarcascade_upperbody.xml')

cap = cv2.VideoCapture('resources/header1.mp4')
# cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    _, img = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    side_faces = side_face_cascade.detectMultiScale(gray, 1.1, 4)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 2)
    for (x, y, w, h) in side_faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)
    for (x, y, w, h) in upper_bodies:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
    # Display
    cv2.imshow('img', img)
    # Stop if escape key is pressed
    k = cv2.waitKey(30) & 0xff
    if k==27:
        break
# Release the VideoCapture object
cap.release()