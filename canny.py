import numpy as np
import cv2

cap = cv2.VideoCapture(0)
kernel_size = 3
scale = 1
delta = 0
ddepth16 = cv2.CV_16S
ddepth8 = cv2.CV_8U
lowThreshold = 0
max_lowThreshold = 100
ratio = 3



def CannyThreshold(lowThreshold):
    detected_edges = cv2.GaussianBlur(gray,(3,3),0)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(frame,frame,mask = detected_edges) 
    cv2.imshow('Canny',dst)

cv2.namedWindow('Canny Trackbar')
cv2.createTrackbar('Min threshold','Canny Trackbar',lowThreshold, max_lowThreshold, CannyThreshold)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_lap = cv2.Laplacian(gray,ddepth16,ksize = kernel_size,scale = scale,delta = delta)
    dst = cv2.convertScaleAbs(gray_lap)
    
    gray_sob = cv2.Sobel(gray,ddepth8,0,1,ksize = 3, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    dst2 = cv2.convertScaleAbs(gray_sob)

    # Display the resulting frame
    cv2.imshow('Original',frame)
    cv2.imshow('Laplacian',dst)
    cv2.imshow('Sobel',dst2)

    CannyThreshold(lowThreshold) 
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
