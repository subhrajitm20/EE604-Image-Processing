from matplotlib import pyplot
import cv2
import numpy as np
# import cv2 as cv
import sys
from PIL import Image

station_number = sys.argv[1]
# Creating a black background to place the white circles
blank_image = np.zeros((300,500,3), np.uint8)


radius = 25
color = (255, 255, 255)
thickness = -1

i=9
if(len(station_number) == 2):
	f = int(station_number[0])
	s = int(station_number[1])
if(len(station_number) == 1):
	f = 0
	s = int(station_number[0])

for p in range (0,2):
	if(p==0):
		i = f
		off = 0
	else:
		i = s
		off = 260

	co = (0,0,0)
	if i==0 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)
				
		
		c1 = (125+off, 85)
		c2 = (125+off, 140)
		c3 = (125+off, 195)
		image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		image = cv2.circle(blank_image, c3, radius, co, thickness)

	if i==1 :
		for y in range(30,271,55):
			x = 140

			center = (x+off, y)
			image = cv2.circle(blank_image, center, radius, color, thickness)


	if i==2 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)

		c1 = (70+off,85)
		c2 = (125+off,85)
		c3 = (125+off, 195)
		c4 = (180+off, 195)
		image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

	if i==3 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)


		
		c1 = (70+off,85)
		c2 = (125+off,85)
		c3 = (70+off, 195)
		c4 = (125+off, 195)
		image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

	if i==4 :
		for x in range(70, 210, 55):

			center = (x+off, 140)
			image = cv2.circle(blank_image, center, radius, color, thickness)

		for y in range (30, 271, 55):
			center = (180+off, y)
			image = cv2.circle(blank_image, center, radius, color, thickness)
		
		for y in range (30, 141, 55):
			center = (70+off, y)
			image = cv2.circle(blank_image, center, radius, color, thickness)

	if i==5 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)

		c1 = (180+off,85)
		c2 = (125+off,85)
		c3 = (70+off, 195)
		c4 = (125+off, 195)
		image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

	if i==6 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)

		c1 = (180+off,85)
		c2 = (125+off,85)
		# c3 = (70, 195)
		c4 = (125+off, 195)
		image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		# image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

	if i==7 :

		for y in range (30, 271, 55):
			center = (180+off, y)
			image = cv2.circle(blank_image, center, radius, color, thickness)

		for x in range(70, 210, 55):

			center = (x+off, 30)
			image = cv2.circle(blank_image, center, radius, color, thickness)
		# for y in range(30,271,55):
		# 	for x in range(70, 210, 55):

		# 		center = (x, y)
		# 		image = cv2.circle(blank_image, center, radius, color, thickness)

		# 	for x in range(320, 460, 55):

		# 		center = (x, y)
		# 		image = cv2.circle(blank_image, center, radius, color, thickness)

	if i==8 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)


		# c1 = (180,85)
		c2 = (125+off,85)
		# c3 = (70, 195)
		c4 = (125+off, 195)
		# image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		# image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

	if i==9 :
		for y in range(30,271,55):
			for x in range(70, 210, 55):

				center = (x+off, y)
				image = cv2.circle(blank_image, center, radius, color, thickness)

		# c1 = (180,85)
		c2 = (125+off,85)
		c3 = (70+off, 195)
		c4 = (125+off, 195)
		# image = cv2.circle(blank_image, c1, radius, co, thickness)
		image = cv2.circle(blank_image, c2, radius, co, thickness)
		image = cv2.circle(blank_image, c3, radius, co, thickness)
		image = cv2.circle(blank_image, c4, radius, co, thickness)

im = Image.fromarray(blank_image)
im.save('dotmatrix.jpg')

# Just adding this comment to check the branch feature of github
# Well it's cool
# Adding a new comment after completing steps
# Trying to push changes via terminal