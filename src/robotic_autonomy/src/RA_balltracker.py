#!/usr/bin/env python
import numpy as np
import roslib
#roslib.load_manifest('my_package')
import cv2
import rospy
import sys
import imutils
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

orange_lower = (90,0,110)
orange_upper = (110,255,255)

bridge = CvBridge()

img_ballx = 0
img_bally = 0