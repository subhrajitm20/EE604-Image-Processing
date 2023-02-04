import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Read in the image

path= str(sys.argv[1])
image= cv2.imread(path)

# convert to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

mean_rgb= cv2.mean(image)

a=abs(mean_rgb[1]-mean_rgb[2])
b=abs(2*mean_rgb[1]-mean_rgb[0]-mean_rgb[2])

if int(a) in range(0,80):
  if int(b) in range(12,85):
    m=2
if int(a) in range(0,6):
  if int(b) in range(1,8):
    m=1
if int(a) in range(6,20):
  if int(b) in range(0,12):
    m=3
print(m)