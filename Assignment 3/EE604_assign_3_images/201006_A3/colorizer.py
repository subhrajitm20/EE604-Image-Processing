import cv2
from PIL import Image
from matplotlib import pyplot
import numpy as np
import matplotlib.image as image
import sys

p1 = sys.argv[1]
p2 = sys.argv[2]
p3 = sys.argv[3]

# def convert(im):
#     xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
#     rgb = im.astype(np.float)
#     rgb[:,:,[1,2]] -= 128
#     rgb = rgb.dot(xform.T)
#     np.putmask(rgb, rgb > 255, 255)
#     np.putmask(rgb, rgb < 0, 0)
#     return np.uint8(rgb)

# im = Image.open("Assignment 3/EE604_assign_3_images/Y.jpg")

imgy = cv2.imread(p1, cv2.IMREAD_GRAYSCALE)
imgcr = cv2.imread(p3, cv2.IMREAD_GRAYSCALE)
imgcb = cv2.imread(p2, cv2.IMREAD_GRAYSCALE)

# imgreal = cv2.imread(r"Assignment 3/EE604_assign_3_images/TrueflyingEle.jpg")

imgcr = cv2.pyrUp(imgcr)
imgcb = cv2.pyrUp(imgcb)

imgcr = cv2.pyrUp(imgcr)
imgcb = cv2.pyrUp(imgcb)

(x, y) = imgcr.shape

# nimgy = imgy[:,:,1]
# imgcr = imgcr[:,:,1]
# imgcb = imgcb[:,:,1]

# s1 = imgy[0]
# s2 = imgy[1]
# imgy = np.append()
imgy = np.vstack([imgy, imgy[621]])
imgy = np.vstack([imgy, imgy[620]])           

img_w = cv2.merge([imgy, imgcr, imgcb])


# imageYCbCr = imgy
# imageYCbCr[:,:,1] = imgy
# imageYCbCr[:,:,2] = imgcb
# imageYCbCr[:,:,3] = imgcr
# img_w = cv2.merge('YCrCb', (imgy, imgcr, imgcb))

# img_w[:,:,0] = nimgy
# img_w[:,:,1] = imgcb
# img_w[:,:,2] = imgcr
fi = cv2.cvtColor(img_w, cv2.COLOR_YCR_CB2BGR)
img_w = cv2.cvtColor(img_w, cv2.COLOR_YCR_CB2RGB)
im2 = 2*img_w

k = imgy.shape
l = imgcb.shape
m = imgcr.shape
n = img_w.shape
# n = imgreal.shape

# # cv2.imshow()
cv2.imwrite("flyingelephant.jpg", fi)
# cv2.waitKey(0)
# cv2.destroyAllWindows

# k = nimgy.shape
# l = imgcb.shape
# m = imgcr.shape
# n = img_w.shape
# n = imgreal.shape
# print(k)
# print(l)
# print(m)
# print(n)
# print(imgy.dtype)
# print(imgcr.dtype)
# print(imgcb.dtype)
# print(imgcb)
# print(img)

# The output 5 and output 8 are different
# output5:

# img_w[:,:,0] = nimgy
# img_w[:,:,1] = imgcb
# img_w[:,:,2] = imgcr

# fi = cv2.cvtColor(img_w, cv2.COLOR_YCR_CB2BGR)

# output8:

# img_w[:,:,0] = nimgy
# img_w[:,:,1] = imgcb
# img_w[:,:,2] = imgcr

# fi = cv2.cvtColor(img_w, cv2.COLOR_YCR_CB2RGB)