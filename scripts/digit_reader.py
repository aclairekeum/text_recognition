#!/usr/bin/env python

""" This is a script that uses computer vision and a machine learning model
    to read handwritten digits. """

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.externals import joblib

class DigitReader(object):
    """ placeholder """

    def __init__(self, image_topic):
        """ Initialize """
        rospy.init_node('digit_reader')

        # Load machine learning model
        self.model = joblib.load('model/model.pkl')

        cv2.namedWindow('video_window')
        # cv2.namedWindow('final')

        self.digit = -1
        self.digit_confidence = 0

        self.bridge = CvBridge()    # used to convert ROS messages to OpenCV
        rospy.Subscriber(image_topic, Image, self.process_image)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def process_image(self, msg):
        """ Process image messages from ROS """
        gray_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Fuzzing to de-noise and edge-detected
        fuzzed = cv2.bilateralFilter(gray_image, 9, 75, 75)
        edged = cv2.Canny(fuzzed, 0, 1)

        # Locate contours and extract those with 4 vertices
        (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
        rect_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Check for four sides and a reasonably large perimeter.
            if len(approx) != 4 or peri < 100:
                continue
            # Order vertices as top-left, top-right, bottom-right, bottom-left
            pts = approx.reshape(4, 2)
            rect = np.zeros((4, 2), dtype = "float32")
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
                np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2)),
                np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2)),
                np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            ))
            r_height = r_width
            dst = np.array(
                [
                    [0, 0],
                    [r_width - 1, 0],
                    [r_width - 1, r_height - 1],
                    [0, r_height - 1]
                ],
                dtype = "float32"
            )
            # See if the rectangle is square. Gave a tolerance of 30% of the max for each sides.
            # Then draw contour, pass it to the next step. 
            if abs(r_width - r_height) < 0.3 * max(r_width, r_height):
                rect_cnt = approx
                cv2.drawContours(gray_image, [rect_cnt], -1, 255, 3)
                break

        cv2.imshow("video_window", gray_image)
        # If no square was found, return
        if rect_cnt is None:
            return

        # Warp the image to un-perspective-ify the square
        M = cv2.getPerspectiveTransform(rect, dst)
        warp = cv2.warpPerspective(gray_image, M, (r_width, r_height))

        # Crop out the border
        border_ratio = 0.05
        height, width = warp.shape
        h_border = height * border_ratio
        w_border = width * border_ratio
        cropped = warp[h_border:height-h_border, w_border:width-w_border]

        # Invert to match training data and drop the lower end to 0 (instead of a dark-ish gray)
        inverted = (255 - cropped)
        ret, thresholded = cv2.threshold(inverted, 195, 255, cv2.THRESH_TOZERO)
        cv2.imshow("final", thresholded)

        # Resize to match standard size
        resized = cv2.resize(thresholded, (8, 8))

        # Use model to predict the digit
        data = resized.reshape((1,64)) / 16.0 # Divide by 16 and round to integers to match training data
        data.round()
        self.digit = self.model.predict(data)[0]
        self.digit_confidence = self.model.predict_proba(data).max()

        cv2.waitKey(1)

    def run(self):
        """ The main run loop, publish twist messages """
        r = rospy.Rate(5)
        my_twist = Twist()
        while not rospy.is_shutdown():
            print self.digit, self.digit_confidence
            if self.digit == 4:
                my_twist.angular.z = 0.1
            elif self.digit == 0:
                my_twist.angular.z = -0.1
            else:
                my_twist.angular.z = 0
            self.pub.publish(my_twist)
            r.sleep()

if __name__ == '__main__':
    node = DigitReader('/camera/image_raw')
    node.run()
