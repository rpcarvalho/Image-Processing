import cv2
import numpy as np
from PIL import ImageGrab

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('people-walking.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()

def close():
    cap.release()
    cv2.destroyAllWindows()
    
def drawCountoursBox(original, edit, minSize = -1):
    img = original.copy()
    im2, contours, hierarchy = cv2.findContours(edit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(frame, contours, -1, (0,255,0), 2)
    cv2.imshow('cnt',im2)
    for cnt in contours:
        #x,y,w,h = cv2.boundingRect(cnt)
        #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        area = rect[1][0] * rect[1][1]
        if minSize != -1:
            if area > minSize:
                cv2.drawContours(img,[box],0,(0,0,255),2)
        else:
            cv2.drawContours(img,[box],0,(0,0,255),2)
    return img, contours

def noiseReduction(erode = 1, dilate = 1, blur = 0):
    #morph transf
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(fgmask, kernel, iterations = erode)
    dilation = cv2.dilate(erosion, kernel, iterations = dilate)
    if blur > 0:
        blured = cv2.medianBlur(dilation, 1)
        return blured
    return dilation

def screenCapture():
    frame = np.array(ImageGrab.grab(bbox=(0,100,750,480)))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame
    
while(True):
    ret, frame = cap.read()
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    fgmask = fgbg.apply(frame)    
    
    nonoise = noiseReduction(1,10)
    res, cnt = drawCountoursBox(frame, nonoise, 3000)
    
    #res = cv2.bitwise_and(frame, frame, mask= blur)
    
    #cv2.imshow('frame',frame)
    #cv2.imshow('fgmask',fgmask)
    #cv2.imshow('Dilation',nonoise)
    cv2.imshow('res',res)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
