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
    blur = cv2.blur(gray,(25,25))
    edges = cv2.Canny(blur,lowThreshold*5,lowThreshold*10,3) 
    drawing = np.zeros(frame.shape,np.uint8)
    contours,hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE, (5,5))
    for cnt in contours:
        color = np.random.randint(0,255,(3)).tolist()  # Select a random color
        cv2.drawContours(drawing,[cnt],0,color,2)
        cv2.imshow('output',drawing)


cv2.namedWindow('Canny Trackbar')
cv2.createTrackbar('Min threshold','Canny Trackbar',lowThreshold, max_lowThreshold, CannyThreshold)


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_lap = cv2.Laplacian(gray,ddepth16,ksize = kernel_size,scale = scale,delta = delta)
    dst = cv2.convertScaleAbs(gray_lap)
    
    gray_sob = cv2.Sobel(gray,ddepth8,0,1,ksize = -1, scale = scale, delta = delta, borderType = cv2.BORDER_DEFAULT)
    dst2 = cv2.convertScaleAbs(gray_sob)

    bllap = cv2.GaussianBlur(gray_lap,(25,25),0)
    dst3 = cv2.convertScaleAbs(bllap)


    

    # Display the resulting frame
    # cv2.imshow('Origin',frame)
    cv2.imshow('Laplacian',dst)
    cv2.imshow('Scharr',dst2)
    cv2.imshow('Blured Laplacian',dst)

    CannyThreshold(lowThreshold) 
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
