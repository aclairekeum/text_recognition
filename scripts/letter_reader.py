#!/usr/bin/env python

""" This is a script that uses computer vision and a machine learning model
    to read letters. """

import rospy
import rospkg
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque
from sklearn.externals import joblib

class LetterReader(object):
    """ Node for reading signs with letters (F, L, or R) on them """

    def __init__(self, image_topic):
        """ Initialize """
        rospy.init_node('letter_reader')

        # Load machine learning model
        path = rospkg.RosPack().get_path('text_recognition')
        self.model = joblib.load('{}/lettermodel/lettermodel.pkl'.format(path))
        self.target_letters = 'FLR' # key for the targets

        # Bridge to convert ROS messages to OpenCV
        self.bridge = CvBridge()

        # To filter out noise when there is not a letter in the frame,
        # we only accept predictions when the confidence is high enough.
        self.min_letter_confidence = 0.6

        # We also use a sliding window and only accept the prediction when there is agreement
        self.max_len = 5
        self.last_letters = deque([-1], maxlen=self.max_len)

        # Subscribe to the image topic
        rospy.Subscriber(image_topic, Image, self.process_image)

    def process_image(self, msg):
        """ Process image messages from ROS """
        # Receive grayscale image
        gray_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Fuzz to de-noise and edge-detect
        fuzzed = cv2.bilateralFilter(gray_image, 9, 75, 75)
        edged = cv2.Canny(fuzzed, 0, 1)

        # Locate contours and extract those with 4 vertices
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        dst = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Check for four sides and a reasonably large perimeter.
            if len(approx) != 4 or peri < 100:
                continue
            # Order vertices as top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype = 'float32')
            center = pts.mean(axis = 0)
            try:
                rect[:2] = sorted([pt for pt in pts if pt[1] < center[1]], key = lambda p: p[0])
                rect[2:] = sorted([pt for pt in pts if pt[1] >= center[1]], key = lambda p: p[0], reverse = True)
            except:
                continue
            # Calculate dimensions
            tl, tr, br, bl = rect
            r_width = int(max(
                np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2)),
                np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            ))
            r_height = int(max(
                np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)),
                np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            ))
            # See if the rectangle is square. Gave a tolerance of 30% of the max for each sides.
            # Then draw contour, create destination array, and pass it to the next step. 
            if abs(r_width - r_height) < 0.3 * max(r_width, r_height):
                cv2.drawContours(gray_image, [approx], -1, 255, 3)
                sq_side = max(r_width, r_height)
                dst = np.array(
                    [
                        [0, 0],
                        [sq_side - 1, 0],
                        [sq_side - 1, sq_side - 1],
                        [0, sq_side - 1]
                    ],
                    dtype = 'float32'
                )
                break

        # If no square was found, return
        if dst is None:
            return

        # Warp the image to un-perspective-ify the square
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(gray_image, M, (sq_side, sq_side))

        # Crop out the border
        border_ratio = 0.17 #0.07
        height, width = warp.shape
        h_border = int(height * border_ratio)
        w_border = int(width * border_ratio)
        cropped = warp[h_border:height-h_border, w_border:width-w_border]

        # Invert to match training data and drop the lower end to 0 (instead of a dark-ish gray)
        inverted = (255 - cropped)
        ret, thresholded = cv2.threshold(inverted, 195, 255, cv2.THRESH_TOZERO)

        # Resize to match standard size
        resized = cv2.resize(thresholded, (8, 8))

        # Use model to evaluate the highest confidence in letter prediction
        # Flatten image to match training data
        data = resized.ravel()
        self.letter_confidence = self.model.predict_proba(data).max()
        print  self.model.predict_proba(data)
        # Only accept the prediction if it has a high enough confidence; use -1 if not
        # Push the result to the deque of the last N results
        letter = -1
        if self.letter_confidence > self.min_letter_confidence:
            letter = self.model.predict(data)[0]

        self.last_letters.append(letter)

        # Show guess on the stream image
        cv2.putText(gray_image, self.target_letters[int(letter)], (10,60), cv2.FONT_HERSHEY_SIMPLEX, 2, 0, 2)

        # Show the stream and final extracted images
        # The stream will have a white contour drawn on it showing the extracted square
        # (or not if no square was found) and the predicted letter in the top left corner
        cv2.imshow('stream', gray_image)
        cv2.imshow('final', thresholded)
        cv2.waitKey(1)

    def run(self):
        """ The main run loop, print letters """
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
            # If the last N letters were the same, accept that letter as correct
            letter = self.last_letters[-1]
            if letter != -1 and self.last_letters.count(letter) == self.max_len:
                print self.target_letters[int(letter)]
            r.sleep()

if __name__ == '__main__':
    node = LetterReader('/camera/image_raw')
    node.run()
