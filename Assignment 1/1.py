# import cv2
# from PIL import Image

# # from matplotlib import image
# from matplotlib import pyplot
# import cv2 
# import numpy as np

# # Creating a black background to place the white circles
# blank_image = np.zeros((300,500,3), np.uint8)



# # Radius of circle
# radius = 5

# # Red color in BGR
# color = (255, 255, 255)

# # Line thickness of -1 px
# # thickness = -1

# i=0

# if i==0 :
# 	for y in range(30,271):
# 		for x in range(30, 221):

# 			# Center coordinates
# 			center_coordinates = (x, y)
# 			image = cv2.circle(blank_image, center_coordinates, radius, color)
# 			x = x + 55

# 		for x in range(280, 471):

# 			# Center coordinates
# 			center_coordinates = (x, y)
# 			image = cv2.circle(blank_image, center_coordinates, radius, color)
# 			x = x + 55
# 		y = y + 55

# pyplot.imshow(image)
# pyplot.show()




# Python program to explain cv2.circle() method
   
# importing cv2
import cv2
import numpy as np
from matplotlib import pyplot

black_image = np.zeros((300,500,3), np.uint8)
# Center coordinates
center_coordinates = (120, 100)
 
# Radius of circle
radius = 25
  
# Red color in BGR
color = (255, 255, 255)

# Line thickness of -1 px
thickness = -1

# Using cv2.circle() method
# Draw a circle of red color of thickness -1 px
image = cv2.circle(black_image, center_coordinates, radius, color, thickness)
  

center_coordinates = (240, 100)
 
# Radius of circle
radius = 25
  
# Red color in BGR
color = (255, 255, 255)

# Line thickness of -1 px
thickness = -1

# Using cv2.circle() method
# Draw a circle of red color of thickness -1 px
image = cv2.circle(black_image, center_coordinates, radius, color, thickness)
# Displaying the image
# cv2.imshow('window_name', image)

pyplot.imshow(image)
pyplot.show()