import cv2
import numpy as np
import pyaudio
import threading

p = pyaudio.PyAudio()
volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1   # in seconds, may be float
f = 440
stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=fs,
                    output=True)

cap = cv2.VideoCapture(0) #webcam
fgbg = cv2.createBackgroundSubtractorMOG2() #movement
ball = [0,0,0,0]
old_ball = ball

def close():
    cap.release()
    cv2.destroyAllWindows()
    p.close(stream)

def noiseReduction(mask, erode = 1, dilate = 1, blur = 0):
    #morph transf
    kernel = np.ones((5,5),np.uint8)
    erosion = cv2.erode(mask, kernel, iterations = erode)
    dilation = cv2.dilate(erosion, kernel, iterations = dilate)
    if blur > 0:
        blured = cv2.medianBlur(dilation, 1)
        return blured
    return dilation

def colorFilter(hsvFrame):
    lower_color = np.array([60,80,120])
    upper_color = np.array([120,235,230])
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
    print('auid')
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
    
    #COLOR FILTER
    colorFil = colorFilter(hsv)
    #res = cv2.bitwise_and(frame,frame, mask= mask)
    bits = cv2.bitwise_and(colorFil, colorFil, mask= nonoise)

    #nonoise = noiseReduction(colorFilter, 5, 4)
    #res, ret = drawCountoursBox(frame, colorFil, 3000, True)
    #res, ret = drawCountoursBox(frame, bits, 3000, True)

    ball = findObject(bits, 3000, 6000, True)
    if np.any(ball):
        x,y,w,h = ball  
        old_ball = ball
    else:
        x,y,w,h = old_ball
    #cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
    cv2.line(frame, (x,y+int(h/2)), (x+w,y+int(h/2)), (255,0,0))
    cv2.line(frame, (x+int(w/2),y), (x+int(w/2),y+h), (255,0,0))

    f = 220 + int((y+h)*1.4)
    string = 'ball:' + str(f)
    cv2.putText(frame, string, (50,50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4.0, (255,0,0), 3)

    cv2.imshow('Ball',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
