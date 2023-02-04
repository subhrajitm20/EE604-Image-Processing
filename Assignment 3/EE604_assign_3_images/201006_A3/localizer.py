import cv2
from PIL import Image
from matplotlib import pyplot
import numpy as np
import matplotlib.image as image
import sys

p = sys.argv[1]

img = cv2.imread(p, cv2.IMREAD_COLOR)

# cv2.imshow("img", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# im = Image.open("Assignment 3/EE604_assign_3_images/location.png")
# gb = abs(img[12:75,40:100,1]-img[12:75,40:100,0])
# img_matrix2 = abs((2*img[12:75,40:100,1])-img[12:75,40:100,0]-img[12:75,40:100,2])
# img = img[12:75,40:100]
gb = abs(img[:,:,1]-img[:,:,2])
# rg = abs(img[:,:,2]-img[:,:,1])
# rb = abs(img[:,:,2]-img[:,:,0])
# rgb = abs((img[:,:,1]+img[:,:,0]+img[:,:,2])/3)
ggrb = abs((2*img[:,:,1])-img[:,:,0]-img[:,:,2])

(x,y,z) = img.shape
gbm=0
ggrbm = 0
for i in range (0,x):
    for j in range (0,y):
        gbm += abs(int(img[i][j][1]) - int(img[i][j][2]))
        ggrbm += abs(int((2*img[i][j][1])) - int(img[i][j][2]) - int(img[i][j][0]))

gbm = gbm/(x*y)
ggrbm = ggrbm/(x*y)

gba = np.mean(gb)
# rga = np.mean(rg)
# rba = np.mean(rb)
# rgba = np.mean(rgb)
ggrba = np.mean(ggrb)

# img_matrix = image.imread("Assignment 3/EE604_assign_3_images/location.png")
# k = img.shape
# print(gb)
# img_matrix = img[]
# print(img_matrix)


# grass = 0
# road = 0
# building = 0
# for i in range (img.shape[0]):
#     for j in range (img.shape[1]):
#         if(ggrb[i][j]>=12 & ggrb[i][j]<=85):
#             grass = grass+1
#         if(gb[i][j]>=0 & gb[i][j]<=6):
#             road = road+1
#         if(gb[i][j]>=6 & gb[i][j]<=20):
#             building = building+1

# # if(grass>=road & grass>=building):
# #     print(2)
# if(road>=grass & road>=building):
#     print(3)
# elif(grass>=road & grass>=building):
#     print(2)
# else:
#     print(1)

# print(gba)
# # print(rga)
# # print(rba)
# # print(rgba)
# print(ggrba)

# print(gbm)
# print(ggrbm)

if(gbm>=0 and gbm<=80 and ggrbm>=12 and ggrbm<=85):
    print(2)    #Grass
if(gbm >=0 and gbm<=6 and ggrbm>=1 and ggrbm<=8):
    print(1)    #Building
if(gbm>=6 and gbm<=20 and ggrbm >=0 and ggrbm<=12):
    print(3)    #Road
# else:
#     print("Bahut dukh hua")