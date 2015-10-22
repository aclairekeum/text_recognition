import cv2
from sys import argv

def nothing(x):
    pass

cv2.namedWindow('slider_window')
blue_lower_bound = 95
cv2.createTrackbar('blue lower bound', 'slider_window', 95, 255, nothing)
blue_upper_bound = 151
cv2.createTrackbar('blue upper bound', 'slider_window', 151, 255, nothing)
green_lower_bound = 127
cv2.createTrackbar('green lower bound', 'slider_window', 127, 255, nothing)
green_upper_bound = 255
cv2.createTrackbar('green upper bound', 'slider_window', 255, 255, nothing)
red_lower_bound = 0
cv2.createTrackbar('red lower bound', 'slider_window', 0, 255, nothing)
red_upper_bound = 38
cv2.createTrackbar('red upper bound', 'slider_window', 38, 255, nothing)


im = cv2.imread(argv[1])
cv2.imshow('original', im)

while True:
    bl = cv2.getTrackbarPos('blue lower bound', 'slider_window')
    bu = cv2.getTrackbarPos('blue upper bound', 'slider_window')
    gl = cv2.getTrackbarPos('green lower bound', 'slider_window')
    gu = cv2.getTrackbarPos('green upper bound', 'slider_window')
    rl = cv2.getTrackbarPos('red lower bound', 'slider_window')
    ru = cv2.getTrackbarPos('red upper bound', 'slider_window')

    binary = cv2.inRange(im, (bl, gl, rl), (bu, gu, ru))
    cv2.imshow('binary', binary)
    cv2.waitKey(0)
