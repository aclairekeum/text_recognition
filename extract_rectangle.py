import cv2
import numpy as np
from sys import argv
from sklearn.externals import joblib

# Much of the rectangle extraction is taken from here:
# http://www.pyimagesearch.com/2014/04/21/building-pokedex-python-finding-game-boy-screen-step-4-6/

def invertImg(imagem):
    imagem = (255-imagem)
    return imagem

# Read an image file in images folder.
im = cv2.imread(argv[1])
# cv2.imshow("original", im)

# Save a grayscale copy of the original
gray_orig = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

# Fuzzing to denoise
fuzzed = cv2.bilateralFilter(im, 9, 75, 75)

# BGR filtering, might not work as well for actual pictures in actual lighting conditions
# lower_bounds = (0, 0, 49)
# upper_bounds = (113, 69, 255)
# binary = cv2.inRange(fuzzed, lower_bounds, upper_bounds)
# cv2.imshow("binary", binary)

binary= gray_orig

# Edge detection
edged = cv2.Canny(binary, 0, 1)
cv2.imshow("edged", edged)

# Locate contours and extract those with 4 vertices
(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)



rect_cnt = None
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    
    # Check for four sides and a reasonably large perimeter
    if len(approx) == 4 and peri > 300:
        # Order vertices as top-left, top-right, bottom-right, bottom-left
        pts = approx.reshape(4, 2)
        rect = np.zeros((4, 2), dtype = "float32")
        center = pts.mean(axis = 0)
        rect[:2] = sorted([pt for pt in pts if pt[1] < center[1]], key = lambda p: p[0])
        rect[2:] = sorted([pt for pt in pts if pt[1] >= center[1]], key = lambda p: p[0], reverse = True)

        # Calculate size of result, might be able to use this in the filtering of rectangles (by using a known h/w ratio)
        tl, tr, br, bl = rect
        width = int(max(
            np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)),
            np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        ))
        height = int(max( 
            np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)),
            np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        ))
        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ],
            dtype = "float32"
        )
        print "width = ", width, "height = ",height

        if abs(width-height)<0.3*max(width, height):
            rect_cnt = approx
            cv2.drawContours(im, [rect_cnt], -1, (0, 255, 0), 3)
            break

if rect_cnt is None:
    print "no square contour found"
    exit(0)
cv2.imshow("with square", im)



# Warp the image to un-perspective-ify the rectangle
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(gray_orig, M, (width, height))
cv2.imshow("warped", warp)

# Crop out the border
border_ratio = 0.05
height, width = warp.shape
h_border = height * border_ratio
w_border = width * border_ratio
crop = warp[h_border:height-h_border, w_border:width-w_border]
cv2.imshow("cropped", crop)

# Invert to match training data
inverted = invertImg(crop)
cv2.imshow("inverted", inverted)

# Drop the lower end to 0 (instead of a dark-ish gray)
ret, thresholded = cv2.threshold(inverted,200,255, cv2.THRESH_TOZERO)
cv2.imshow("thresholded", thresholded)
# cv2.imwrite("five.png", inverted)

# Resize to match standard size
resized = cv2.resize(thresholded, (8, 8))

# Use model to predict the digit
model = joblib.load("model/model.pkl")
data = resized.reshape((1,64)) / 16.0 # Divide by 16 and round to integers to match training data
data.round()
print model.predict(data)

cv2.waitKey()
