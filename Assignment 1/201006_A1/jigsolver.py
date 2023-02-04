import cv2
from PIL import Image

# from matplotlib import image
from matplotlib import pyplot
import numpy as np
import sys
path = sys.argv[1]

im = Image.open(path)
img_matrix = im.load()

left = 0
top = 0
right = 190
bottom = 200

im1 = im.crop((left, top, right, bottom))  # paint and bring down
im1_ar = np.array(im1)
im2 = im.crop((0,200,190,410)) # Move up
im2_ar = np.array(im2)
im3 = im.crop((515, 150, 700, 330))  # Mirror at right
im3_ar = np.array(im3)
im4 = im.crop((369, 369, 797, 421)) # Mirror at top
im4_ar = np.array(im4)
# Shows the image in image viewer

# print(im1_ar.shape)
# img = cv2.imread(im2)
# img1 = img[:, :, [2, 1, 0]]
r, g, b = im1.split()
im1 = Image.merge('RGB', (r, b, g))

im3 = im3.transpose(Image.FLIP_LEFT_RIGHT)
im4 = im4.transpose(Image.FLIP_TOP_BOTTOM)
im2 = im2.transpose(Image.FLIP_TOP_BOTTOM)

im.paste(im1, (0, 200))
im.paste(im2, (0, 0))
im.paste(im3, (515, 150))
im.paste(im4, (369, 369))


for x in range (184, 192):
    for y in range (0, 160):
        r1 = img_matrix[182,y][0]
        g1 = img_matrix[182,y][1]
        b1 = img_matrix[182,y][2]
        r2 = img_matrix[192,y][0]
        g2 = img_matrix[192,y][1]
        b2 = img_matrix[192,y][2]
        img_matrix[x,y] = (int((r1+r2)/2), int((g1+g2)/2), int((b1+b2)/2))



for x in range (188, 192):
    for y in range (245, 380):
        r1 = img_matrix[182,y][0]
        g1 = img_matrix[182,y][1]
        b1 = img_matrix[182,y][2]
        r2 = img_matrix[192,y][0]
        g2 = img_matrix[192,y][1]
        b2 = img_matrix[192,y][2]
        img_matrix[x,y] = (int((r1+r2)/2), int((g1+g2)/2), int((b1+b2)/2))

for x in range (0,190):
    for y in range (400, 405):
        img_matrix[x,y] = img_matrix[x,399]

    for y in range (405, 410):
        img_matrix[x,y] = img_matrix[x, 410]

# pyplot.imshow(im)
# pyplot.show()

# cv2.imwrite('jigsolved.jpg', im)
im = im.save('jigsolved.jpg')