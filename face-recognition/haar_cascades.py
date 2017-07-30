import cv2
import numpy as np

webcam = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def close():
    webcam.release()
    cv2.destroyAllWindows()

def firstface(img, f):
    if len(f) > 0:
        (x,y,w,h) = f[0]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

while(True):
    ret, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #print first face
    firstface(frame, faces)

    #for (x,y,w,h) in faces:
     #   cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img', frame)
            
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
