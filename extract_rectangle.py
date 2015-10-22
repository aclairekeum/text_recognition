import cv2
import numpy as np
from sys import argv

im = cv2.imread(argv[1])
cv2.imshow("original", im)

# Fuzzing to denoise
fuzzed = cv2.bilateralFilter(im, 9, 75, 75)
cv2.imshow("fuzzed", fuzzed)

# BGR filtering, might not work as well for actual pictures in actual lighting conditions
lower_bounds = (0, 0, 120)
upper_bounds = (75, 100, 255)
binary = cv2.inRange(fuzzed, lower_bounds, upper_bounds)
cv2.imshow("binary", binary)

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

    if len(approx) == 4:
        rect_cnt = approx
        break
cv2.imshow("with rectangle", im)

if rect_cnt is None:
    print 'no rectangle contour found'
    exit(0)

# Order vertices as top-left, top-right, bottom-right, bottom-left
pts = rect_cnt.reshape(4, 2)
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

# Warp the image to un-perspective-ify the rectangle
M = cv2.getPerspectiveTransform(rect, dst)
warp = cv2.warpPerspective(binary, M, (width, height))
cv2.imshow("warped", warp)

# Resize to match standard size
resized = cv2.resize(warp, (8, 8))
cv2.imshow("resized", resized)

cv2.waitKey()
