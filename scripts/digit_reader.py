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

        self.cv_image = None        # the latest image from the camera
        self.bridge = CvBridge()    # used to convert ROS messages to OpenCV

        cv2.namedWindow('video_window')

        # Load machine learning model
        self.model = joblib.load('model/model.pkl')

        rospy.Subscriber(image_topic, Image, self.process_image)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)

    def process_image(self, msg):
        """ Process image messages from ROS """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        cv2.imshow('video_window', self.cv_image)
        cv2.waitKey(2)

    def run(self):
        """ The main run loop, publish twist messages """
        r = rospy.Rate(5)
        my_twist = Twist()
        while not rospy.is_shutdown():
            my_twist.angular.z = 0
            self.pub.publish(my_twist)
            r.sleep()

if __name__ == '__main__':
    node = DigitReader('/camera/image_raw')
    node.run()
