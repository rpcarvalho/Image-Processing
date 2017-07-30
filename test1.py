import cv2
import numpy as np

webcam = cv2.VideoCapture(0)

def close():
    webcam.release()
    cv2.destroyAllWindows()
    

while(True):
    ret, frame = webcam.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #COLOR FILTER
    lower_color = np.array([35,90,100])
    upper_color = np.array([150,200,250])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    #BLUR
    median = cv2.medianBlur(res,15)
    
    #cv2.imshow('frame',hsv)
    #cv2.imshow('gray',gray)
    #cv2.imshow('mask',mask)
    #cv2.imshow('res',res)
    cv2.imshow('median',median)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
