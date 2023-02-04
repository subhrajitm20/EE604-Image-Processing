from PIL import Image
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import sys
import argparse
from PIL import ImageEnhance

shubh = argparse.ArgumentParser()
shubh.add_argument("im", help = "gandhi")
args = shubh.parse_args()
img = cv.imread(args.im)
path = args.im
path = path[::-1]
x = ""
i =-1
count = 0
while (count!=2):
    x+=path[i]
    if (path[i]=='c'):
        count+=1
# print(x[::-1])

def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv.LUT(src, table)

def hist(img):
    z = np.array(img)
    # w = z.shape[0]
    # h = z.shape[1]
    h,w = img.shape
    sz = w*h

    pixels = []
    for x in range (0,256):
        pixels.append(0)
    
    cnts = []

    
    for l in range(h):
        for b in range(w):
            pixels[z[l, b]]+=1

    pixels/=np.sum(pixels)
    # print(pixels)


    cdf = []
    # cdf[0] = pixels[0]
    sum = 0
    for i in range (0,256):
        sum+=pixels[i]
        cdf.append(sum)
        # cdf.append(tot)
    # cdf*=255

    cdf = np.round(cdf*255,0)

    imgeq = np.zeros((h,w))
    for i in range (imgeq.shape[0]):
        for j in range (imgeq.shape[1]):
            imgeq[i][j] = cdf[img[i][j]]

    return imgeq


img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

if(path[::-1][-5]=='1'):
    img = cv.equalizeHist(img)
    # print(path[::-1][-5])

elif(path[::-1][-5]=='2'):
    cor = hist(img)
    cor = cv.convertScaleAbs(cor, alpha=(255.0))
    img = cv.equalizeHist(img)
    img= img/1.05
    img[210:320, :150] = cor[210:320, :150]

elif(path[::-1][-5]=='3'):
    img = cv.equalizeHist(img)

elif(path[::-1][-5]=='4'):
    t = img
    # t = cv.
    cor = hist(img)
    cor = cv.convertScaleAbs(cor, alpha=(255.0))
    # img = cor
    # converter = ImageEnhance.Brightness(img)
    # img = converter.enhance(0.5)
    # converter = ImageEnhance.Contrast(img)
    img = cv.equalizeHist(img)
    # img[390:480,720:800] = cor[410:500,700:780]
    img[390:500,700:850] = cor[390:500,700:850]

    
# def gammaCorrection(src, gamma):
#     invGamma = 1 / gamma

#     table = [((i / 255) ** invGamma) * 255 for i in range(256)]
#     table = np.array(table, np.uint8)

#     return cv.LUT(src, table)


# img3 = cv.cvtColor(img3, cv.COLOR_BGR2RGB)
# img1 = cv.blur(img1,(3,3))      # Averaging filter/Box filter - Could not do
# img1 = cv.medianBlur(img1,3)      # Median filtering
# img2_new = cv.GaussianBlur(img2, (3,3), 5, 7)
# img1 = cv.bilateralFilter(img1,2,75,75)

# hist,bins = np.histogram(img1.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# plt.plot(cdf_normalized, color = 'r')
# plt.hist(img1.flatten(),256,[0,256], color = 'b')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()

# img1 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)

# img3 = cv.cvtColor(img3, cv.COLOR_BGR2YUV)
# img3[:,:,0] = cv.equalizeHist(img3[:,:,0])
# img3 = cv.cvtColor(img3, cv.COLOR_YUV2BGR)

# img1 = gammaCorrection(img1, 3)

# img = hist(img)
# img = gammaCorrection(img, 0.5)

# hist,bins = np.histogram(img1.flatten(),256,[0,256])
# cdf = hist.cumsum()
# cdf_normalized = cdf * float(hist.max()) / cdf.max()
# plt.plot(cdf_normalized, color = 'b')
# plt.hist(img1.flatten(),256,[0,256], color = 'r')
# plt.xlim([0,256])
# plt.legend(('cdf','histogram'), loc = 'upper left')
# plt.show()

# print("enhanced-"+path[::-1])
# img = cv.convertScaleAbs(img, alpha=(255.0))
cv.imwrite("enhanced-"+path[::-1], img)
# cv.imshow('equ1.png',img)
# cv.waitKey(0)
# cv.destroyAllWindows()
