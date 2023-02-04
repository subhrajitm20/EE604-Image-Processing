from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot
import sys
import argparse

shubh = argparse.ArgumentParser()
shubh.add_argument("im", help = "gandhi")
args = shubh.parse_args()
img1 = cv2.imread(args.im)

def gaussian_kernel(size,sigma):
    c = (int)(size/2)
    k = np.zeros((size,size))
    s = 0
    for x in range (size):
        for y in range (size):
            ss = sigma*sigma
            ss = 2*ss
            k[x][y] = np.exp(-((x*x) + (y*y))/ss)
            # diff = (x-c)**2 + (y-c)**2
            # k[x][y] = np.exp(-diff/ss)
            s+=k[x][y]
    
    for x in range (size):
        for y in range (size):
            k[x][y]/=s

    return k

def gaussian_blur(img, size, sigma):
    ker = [[1]*size]*size
    ker = gaussian_kernel(size, sigma)
    img = cv2.filter2D(img,-1,ker)
    return img
    

# path1 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters1.JPG"
# path2 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters2.JPG"
# path3 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters3.JPG"

# img1 = cv2.imread(path1)
# img2 = cv2.imread(path2)
# img3 = cv2.imread(path3)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img10 = gaussian_blur(img1, 51, 27)
# img20 = cv2.blur(img2,(11,11))
img1 = img1/img10
# img1 = img1.astype("uint8")
# ret, thresh1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img = cv2.convertScaleAbs(img1, alpha=(255.0))
cv2.imwrite('cleaned-gutters.jpg', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
# img1 = cv2.blur(img1,(7,7))      # Averaging filter/Box filter - Could not do
# img1 = cv2.medianBlur(img1,3)      # Median filtering
# img10 = cv2.GaussianBlur(img1, (3,3), 1,1)
# img1 = cv2.bilateralFilter(img1,2,75,75)

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# img1 = cv2.equalizeHist(img1)

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2YUV)
# img1[:,:,0] = cv2.equalizeHist(img1[:,:,0])
# img1 = cv2.cvtColor(img1, cv2.COLOR_YUV2BGR)

# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# ret, thresh1 = cv2.threshold(img1, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# cv2.imshow('IMAGE.jpg', thresh1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

   
# cv2.cvtColor is applied over the
# image input with applied parameters
# to convert the image in grayscale 
# img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
   
# # applying different thresholding 
# # techniques on the input image
# thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
#                                           cv2.THRESH_BINARY, 199, 5)
  
# thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                           cv2.THRESH_BINARY, 199, 5)
  
# # the window showing output images
# # with the corresponding thresholding 
# # techniques applied to the input image
# # cv2.imshow('Adaptive Mean', thresh1)
# cv2.imshow('Adaptive Gaussian', thresh2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# applying different thresholding
# techniques on the input image
# all pixels value above 120 will
# be set to 255
# ret, thresh1 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)
# ret, thresh2 = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY_INV)
# ret, thresh3 = cv2.threshold(img, 200, 255, cv2.THRESH_TRUNC)
# ret, thresh4 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO)
# ret, thresh5 = cv2.threshold(img, 120, 255, cv2.THRESH_TOZERO_INV)
 
# the window showing output images
# with the corresponding thresholding
# techniques applied to the input images

# cv2.imshow('Binary Threshold', thresh1)
# cv2.imshow('Binary Threshold Inverted', thresh2)
# cv2.imshow('Truncated Threshold', thresh3)
# cv2.imshow('Set to 0', thresh4)
# cv2.imshow('Set to 0 Inverted', thresh5)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


# import cv2
# import numpy as np

# path1 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters1.JPG"
# path2 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters2.JPG"
# path3 = "Assignment 2/EE604-Assign2-pictures/EE604-Assign2-pictures/gutters3.JPG"

# img = cv2.imread(path1, -1)

# rgb_planes = cv2.split(img)

# result_planes = []
# result_norm_planes = []
# for plane in rgb_planes:
#     dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
#     bg_img = cv2.medianBlur(dilated_img, 21)
#     diff_img = 255 - cv2.absdiff(plane, bg_img)
#     norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#     result_planes.append(diff_img)
#     result_norm_planes.append(norm_img)

# result = cv2.merge(result_planes)
# result_norm = cv2.merge(result_norm_planes)

# cv2.imwrite('shadows_out.png', result)
# cv2.imwrite('shadows_out_norm.png', result_norm)

# cv2.imshow('IMAGE.jpg', result_norm)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# image = cv2.imread('GFG.png')
   
# # Apply log transformation method
# c = 255 / np.log(1 + np.max(img1))
# log_image = c * (np.log(img1 + 1))
   
# # Specify the data type so that
# # float value will be converted to int
# log_image = np.array(log_image, dtype = np.uint8)

# cv2.imshow('LOG transform', log_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()