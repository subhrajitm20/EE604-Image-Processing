from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
import argparse

shubh = argparse.ArgumentParser()
shubh.add_argument("im", help = "gandhi")
args = shubh.parse_args()
img = cv2.imread(args.im)
path = args.im
path = path[::-1]
x = ""
for i in range(len(path)):
    if (path[i] == '.'):
        i=i+1
        while (path[i] != 's'):
            x += path[i]
            i += 1
        break
x = int(x[::-1])
    

path1 = "tiles1.JPG"
path2 = "tiles2.JPG"
path5 = "tiles5.JPG"
path8 = "tiles8.JPG"
path11 = "tiles11.JPG"

img = cv2.imread(path2)

img1 = cv2.GaussianBlur(img, (5, 5), 2)

canny_edges = cv2.Canny(img1, 50, 150)

z = np.array(img1)
h = z.shape[0]
w = z.shape[1]

arr = cv2.HoughLines(canny_edges, 1, np.pi/180, 150)

def hough_lines_acc(img, rho_res=1, theta_res=1):

    h = z.shape[0]
    w = z.shape[1]
    diag = np.ceil(np.sqrt(h*h + w*w))
    rhos = np.arange(-diag, diag + 1, rho_res)
    thetas = np.deg2rad(np.arange(-90, 90, theta_res))

    for i in range (len(arr[0])):
        _,theta = arr[0][i]
        thetas = thetas + theta

        rho,_ = arr[0][i]
        rhos+=rho

    y_nz = []
    x_nz = []
    
    for i in range (h):
        for j in range (w):
            if(img[i][j] != 0):
                y_nz.append(i)
                x_nz.append(j)

    H = [[1]*len(rhos)]*len(thetas)
    H = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    
    
    for i in range(len(x_nz)): 
        x = x_nz[i]
        y = y_nz[i]

        for j in range(len(thetas)):
            rho = (x * np.cos(thetas[j]) + y * np.sin(thetas[j]))
            rho+=diag
            rho = int(rho)
            H[rho, j] += 1

    return H, rhos, thetas

def supress_nbd(H1, num_peaks, indicies):
    for i in range(num_peaks):
        idx = np.argmax(H1)
        H1_idx = np.unravel_index(idx, H1.shape)
        indicies.append(H1_idx)

        idx_y, idx_x = H1_idx 
        if (idx_x - (11/2)) < 0: min_x = 0
        else: min_x = idx_x - (11/2)
        if ((idx_x + (11/2) + 1) > H.shape[1]): max_x = H.shape[1]
        else: max_x = idx_x + (11/2) + 1

        if (idx_y - (11/2)) < 0: min_y = 0
        else: min_y = idx_y - (11/2)
        if ((idx_y + (11/2) + 1) > H.shape[0]): max_y = H.shape[0]
        else: max_y = idx_y + (11/2) + 1

        for x in range(int(min_x), int(max_x)):
            for y in range(int(min_y), int(max_y)):
                H1[y, x] = 0

                if (x == min_x or x == (max_x - 1)):
                    H[y, x] = 255
                if (y == min_y or y == (max_y - 1)):
                    H[y, x] = 255
                # if(arr[y,x] != 0):
                #     H[y,x] = arr[y,x]

    

def hough_simple_peaks(H, num_peaks):
    indices =  np.argpartition(H.flatten(), -2)[-num_peaks:]
    indices = []
    supress_nbd(H, num_peaks, indices)
    
    np.squeeze(arr)
    # H = H + dup
    # for i in range (arr.shape[0]):
    #     for j in range (arr.shape[1]):
    #         H[i][j] = arr[i][j]
    return indices, H

def hough_lines_draw(img, indicies, rhos, thetas):
    for pairs in arr:
        # rho = rhos[indicies[i][0]]
        # theta = thetas[indicies[i][1]]
        dup = np.array(pairs[0], dtype=np.float64)
        r, theta = dup
        x0 = np.cos(theta) * r
        y0 = np.sin(theta) * r
        x1 = int(x0 + 1000*(-np.sin(theta)))
        y1 = int(y0 + 1000*(np.cos(theta)))
        x2 = int(x0 - 1000*(-np.sin(theta)))
        y2 = int(y0 - 1000*(np.cos(theta)))

        cv2.line(img, (x1, y1), (x2, y2), (51, 255, 51), 3)



H, rhos, thetas = hough_lines_acc(canny_edges)
indicies, H = hough_simple_peaks(H, 8)
hough_lines_draw(img, indicies, rhos, thetas)


cv2.imwrite("robolin-tiles"+str(x)+".jpg",img)
# cv2.imshow('Major Lines: Manual Hough Transform', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()