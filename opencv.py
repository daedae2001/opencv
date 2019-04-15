import cv2
import sys
import numpy as np
import time


img = cv2.imread("imagen1.jpg")  ## Read image file
#cv2.namedWindow('nombre imagen')        ## create window for display
#cv2.imshow('nombre imagen',img)         ## Show image in the window
print ("size of image: ",img.shape)        ## print size of image
#cv2.waitKey(0)                           ## Wait for keystroke
#cv2.destroyAllWindows()                  ## Destroy all windows

image = cv2.imread('imagen1.jpg') # change image name as you need or give sys.argv[1] to read from command line
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert image to gray

cv2.imwrite('gray_image.jpg',gray_image)   # saves gray image to disk

#cv2.imshow('color_image',image)
#cv2.imshow('gray_image',gray_image)

#cv2.waitKey(0)
#cv2.destroyAllWindows()

img = cv2.imread('imagen1.jpg')

for i  in range(2,90,40):
    mul_img = cv2.multiply(img,np.array([1.0]))                    # mul_img = img*alpha alpha contraste beta brillo
    new_img = cv2.add(mul_img,np.array([0.0+i]))                      # new_img = img*alpha + beta
    cv2.imshow('new_image',new_img)
     
    cv2.waitKey(50)

for i  in range(2,90,40):
    mul_img = cv2.multiply(img,np.array([2.0]))                    # mul_img = img*alpha
    new_img = cv2.add(mul_img,np.array([0.0+i]))                      # new_img = img*alpha + beta
    cv2.imshow('new_image',new_img)
     
    cv2.waitKey(50)
for i  in range(2,90,40):
    mul_img = cv2.multiply(img,np.array([3.0]))                    # mul_img = img*alpha
    new_img = cv2.add(mul_img,np.array([0.0+i]))                      # new_img = img*alpha + beta
    cv2.imshow('new_image',new_img)
     
    cv2.waitKey(50)

cv2.destroyAllWindows()

alpha=0.0
for i in range(0,100,1):
    alpha=alpha+float(i)/100.0
    if 0<=alpha<=1:                        # Check if 0<= alpha <=1
        beta = 1.0 - alpha                 # Calculate beta = 1 - alpha
        gamma = 0.0                        # parameter gamma = 0
        img1 = cv2.imread('imagen1.jpg')
        img2 = cv2.imread('gray_image.jpg')
        dst = cv2.addWeighted(img1,alpha,img2,beta,alpha)  # Get weighted sum of img1 and img2
        #dst = np.uint8(alpha*(img1)+beta*(img2))    # This is simple numpy version of above line. But cv2 function is around 2x faster
        cv2.imshow('dst',dst)
        cv2.waitKey(500)
cv2.destroyAllWindows()
