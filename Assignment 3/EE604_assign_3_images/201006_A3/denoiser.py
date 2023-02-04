import cv2
# from PIL import Image
# from matplotlib import pyplot
import numpy as np
import sys
from scipy import signal, interpolate

# import numpy as np
# from scipy import signal, interpolate

p = sys.argv[1]

def imgproc(image, sigmaspatial, sigmarange):

    h,w = image.shape

    # samplespatial = sigmaspatial
    # samplerange = sigmarange

    flatimage = image.flatten()

    mi = np.amin(flatimage)
    ma = np.amax(flatimage)
    diff = ma - mi

    xp = 3
    zp = 3
    xp = xp**2
    zp = zp**4

    sw = int(round((w - 1) / sigmaspatial) + 7)
    sh = int(round((h - 1) / sigmaspatial) + 7)
    sd = int(round(diff / sigmarange) + 7)

    dataflat = np.zeros(sh * sw * sd)

    (ygrid, xgrid) = np.meshgrid(range(w), range(h))

    dimx = np.around(xgrid / sigmaspatial) + 3
    dimy = np.around(ygrid / sigmaspatial) + 3
    dimz = np.around((image - mi) / sigmarange) +3

    # (ygrid, xgrid) = np.meshgrid(range(w), range(h))

    flatx = dimx.flatten()
    flaty = dimy.flatten()
    flatz = dimz.flatten()

    dim = flatz + flaty * sd + flatx * sw * sd
    dim = np.array(dim, dtype=int)

    dataflat[dim] = flatimage

    data = dataflat.reshape(sh, sw, sd)
    weights = np.array(data, dtype=bool)

    dim = 3
    dep = 5
    hdim = round(1.5)
    hdep = round(2.5)

    (gx, gy, gz) = np.meshgrid(range(3), range(3), range(5))
    gx -= int(hdim)
    gy -= int(hdim)
    gz -= int(hdep)

    gx = gx**2
    gy = gy**2
    gz = gz**2
    su = gx+gy+gz
    gridsqr = su
    # gridsqr = ((gx * gx + gy * gy) / (1 * 1)) \
    #     + ((gz * gz) / (1 * 1))
    kernel = np.exp(-0.5 * gridsqr)

    bw = signal.fftconvolve(weights, kernel, mode='same')
    bw = np.where(bw == 0, -2, bw)

    bd = signal.fftconvolve(data, kernel, mode='same')

    nb = bd
    nb = bd / bw
    nb = np.where(bw < -1, 0, nb)

    (ygrid, xgrid) = np.meshgrid(range(w), range(h))

    dimx = (xgrid / sigmaspatial) + 3
    dimy = (ygrid / sigmaspatial) + 3
    dimz = (image - mi) / sigmarange +3

    return interpolate.interpn((range(nb.shape[0]), range(nb.shape[1]), range(nb.shape[2])), nb, (dimx, dimy, dimz))



img = cv2.imread(p, cv2.IMREAD_COLOR)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.medianBlur(img, 11)
# img1 = cv2.bilateralFilter(img, 9, 300, 100)
if(p=="./noisy1.JPG"):
    img = cv2.bilateralFilter(img, 19, 40, 40)
    
ss = 75
sr = 30
fbfr = imgproc(img[:,:,0], ss, sr)
fbfg = imgproc(img[:,:,1], ss, sr)
fbfb = imgproc(img[:,:,2], ss, sr)

fi = img
fi[:,:,0] = fbfr
fi[:,:,1] = fbfg
fi[:,:,2] = fbfb

# img1 = cv2.bilateralFilter(img, 9, 50, 80)
# # img1 = cv2.bilateralFilter(img1, 9, 1, 100)
# # img1 = cv2.bilateralFilter(img1, 9, 1, 100)

# # img2 = cv2.bilateralFilter(img, 9, 100, 10)
# img3 = cv2.bilateralFilter(img, 51, 80, 80)
# img4 = cv2.bilateralFilter(img, 9, 75, 75)
# cv2.imshow("image1", img1)
# cv2.imshow("image2", img2)
cv2.imwrite("denoised.jpg", fi)



# For noisy1 bilateral values of 51,80,80 is giving best result

# 75, 30 - noisy2
# 