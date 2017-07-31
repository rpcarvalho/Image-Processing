import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def close():
    cap.release()
    cv2.destroyAllWindows()
    

while(True):
    ret, frame = cap.read()
    
    mask = np.zeros(frame.shape[:2],np.uint8)

    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (161,79,150,150)

   # cv2.grabCut(frame,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
   # mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  #  img = img*mask2[:,:,np.newaxis]

    cv2.imshow('median',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

close()
