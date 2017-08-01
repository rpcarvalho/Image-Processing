import cv2
import numpy as np
import pyaudio
import threading

p = pyaudio.PyAudio()
volume = 0.5     # range [0.0, 1.0]
fs = 20000       # sampling rate, Hz, must be integer
duration = 1   # in seconds, may be float
f = 440
stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

cap = cv2.VideoCapture(0) #webcam
fgbg = cv2.createBackgroundSubtractorMOG2() #movement

ball = [0,0,0,0]
old_green = ball
old_yellow = ball

def close():
    cap.release()
    cv2.destroyAllWindows()
    p.close(stream)
    quit()

def noiseReduction(mask, erode = 1, dilate = 1, blur = 0):
    #morph transf
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = erode)
    dilation = cv2.dilate(erosion, kernel, iterations = dilate)
    if blur > 0:
        blured = cv2.medianBlur(dilation, 1)
        return blured
    return dilation

def colorFilter(hsvFrame,color,margin=30):
    lower_color = np.array([color-margin,80,120])
    upper_color = np.array([color+margin,235,230])
    img = cv2.inRange(hsvFrame, lower_color, upper_color)
    return img

def findObject(img, minSize, maxSize, isSquare = False):
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    obj = None
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = w*h
        sqr = abs(w-h)
        if area > minSize and area < maxSize:
            if isSquare and sqr < 20:
                obj = cnt
            elif not isSquare:
                obj = cnt
    return cv2.boundingRect(obj)


def audio():
    while True:
        samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
        stream.write(volume*samples)
    

t = threading.Thread(target=audio)
t.start()




while(True):
    ret, flipit = cap.read()
    frame = cv2.flip(flipit, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    #movement
    fgmask = fgbg.apply(hsv)
    nonoise = noiseReduction(fgmask, 1,1)
    
    #GREEN BALL
    colorFil = colorFilter(hsv,90)
    bits = cv2.bitwise_and(colorFil, colorFil, mask= nonoise)
    ball = findObject(bits, 2000, 7000, True)
    if np.any(ball):
        x,y,w,h = ball
        old_green = ball
    else:
        x,y,w,h = old_green
    cv2.line(frame, (x,y+int(h/2)), (x+w,y+int(h/2)), (0,255,0))
    cv2.line(frame, (x+int(w/2),y), (x+int(w/2),y+h), (0,255,0))
    f = 220 + int((y+h)*1)
    cv2.putText(frame, str(f), (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4.0, (0,255,0), 3)

    #YELLOW BALL
    colorFil = colorFilter(hsv,40,20)
    bits = cv2.bitwise_and(colorFil, colorFil, mask= nonoise)
    ball = findObject(bits, 2000, 7000, True)
    if np.any(ball):
        x,y,w,h = ball
        old_yellow = ball
    else:
        x,y,w,h = old_yellow
    cv2.line(frame, (x,y+int(h/2)), (x+w,y+int(h/2)), (50,255,255))
    cv2.line(frame, (x+int(w/2),y), (x+int(w/2),y+h), (50,255,255))
    volume = round(0.9 - (y+h)/480,2)
    if volume < 0:
        volume = 0
    cv2.putText(frame, str(volume), (50,150), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4.0, (50,255,255), 3)
    

    cv2.imshow('Ball',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
