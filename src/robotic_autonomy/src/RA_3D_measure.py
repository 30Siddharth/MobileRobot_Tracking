#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
from math import pi
import roslib
#roslib.load_manifest('my_package')
import cv2
import rospy
import sys
import imutils
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge, CvBridgeError
import pdb

orange_lower = (13,100,150)
orange_upper = (15,255,255)
bridge = CvBridge()

# @dataclass
# class PixelLoc:
#     x: int
#     y: int

# @dataclass
# class XYZLoc:
#     x: float
#     y: float
#     z: float

class Tracker3D():
    rgbFOV = [65,77]
    depthFOV = [86,57]
    pixToDegree = (np.pi/180)*float(86)/640
    center_pixel = (320,240)

    def __init__(self,img_topic_name="/camera/color/image_raw",depth_topic_name="/camera/depth/image_rect_raw",see_image=False):
        
        self.image_sub = rospy.Subscriber(img_topic_name,Image,self.image_cb)
        self.depth_sub = rospy.Subscriber(depth_topic_name,Image,self.depth_cb)
        self.pos_pub = rospy.Publisher('ballxy',Point,queue_size=10)
        self.viz_pub = rospy.Publisher('ball_maker',Marker,queue_size=10)
        self.ballloc_pixel = [0,0]
        self.ballloc_xyz = [0,0,0]
        self.learning_rate = 0.2 # = 1.0 => no averaging
        self.cv_image = None
        self.depth_image = None
        self.mask = None
        self.d = 0.0
        self.thetax = 0.0
        self.thetay = 0.0
        self.phi = 0.0
        
        # plt.ion()
        # self.fig = plt.figure()
        # # ax = self.fig.add_subplot(111, projection='3d')
        # ax = Axes3D(self.fig)
        # print(type(ax))
        # self.plt_xyz = ax.scatter(self.ballloc_xyz[0],self.ballloc_xyz[1],self.ballloc_xyz[2])
        # plt.show()
        # pdb.set_trace()

        # Wait for messages to be published on image and depth topics
        print("Waiting for image and depth topic")
        rospy.wait_for_message(img_topic_name,Image)
        rospy.wait_for_message(depth_topic_name,Image)
        print("-----> Messages received")

        self.rate = rospy.Rate(20)

    def get_depth(self):
        print(self.cv_image.shape)
        print(self.depth_image.shape)
        xdepth = int(self.depthFOV[0]*float(self.ballloc_pixel[0]-self.center_pixel[0])/self.rgbFOV[0]) + int(640/2)
        ydepth = int(self.depthFOV[1]*float(self.ballloc_pixel[1]-self.center_pixel[1])/self.rgbFOV[1]) + int(480/2)
        xdepth = min(xdepth,639)
        ydepth = min(ydepth,479)

        # xdepth = min(int(86*float(self.ballloc_pixel[0])/65),639)
        # ydepth = min(int(86*float(self.ballloc_pixel[1])/65),479)
        print("xdepth: {}".format(xdepth))
        self.d = 0.001*float(self.depth_image[ydepth][xdepth])

    def get_xyz(self):
        self.get_depth()
        print("depth: {}".format(self.d))
        if self.d < 0.05 or self.d > 20.0:
            return
        print(self.ballloc_pixel)
        self.theta = self.pixToDegree*float(self.ballloc_pixel[0]-self.center_pixel[0])
        self.theta = self.learning_rate*self.theta + (1-self.learning_rate)*self.theta
        x = self.d*np.tan(self.theta) #self.d*np.sin(self.theta)
        print("theta: {}".format(self.theta))
        self.ballloc_xyz[0] = self.learning_rate*x + (1-self.learning_rate)*x
        self.ballloc_xyz[1] = self.learning_rate*self.d + (1-self.learning_rate)*self.ballloc_xyz[1]

    def pub_xy(self):
        msg = Point()
        msg.x = self.ballloc_xyz[0]
        msg.y = self.ballloc_xyz[1]
        self.pos_pub.publish(msg)

    def pub_viz(self):
        msg = Marker()
        msg.type = msg.SPHERE
        msg.color.r = 1.0
        msg.color.a = 1.0
        msg.scale.x = 0.2
        msg.scale.y = 0.2
        msg.scale.z = 0.2
        msg.header.frame_id = "/camera_link"
        msg.pose.position.x = self.ballloc_xyz[0]
        msg.pose.position.y = self.ballloc_xyz[1]
        self.viz_pub.publish(msg)

    def viz_3d(self):
        self.plt_xyz.set_offsets(self.ballloc_xyz)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def image_cb(self,data):
        try:
		    self.cv_image = bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError as e:
            print(e)

        cv_image_hsv = cv2.cvtColor(self.cv_image,cv2.COLOR_BGR2HSV)
	    #(rows,cols,channels) = cv_image.shape
        # cv2.circle(self.cv_image,(320,240),10,(255,0,0))

	    #blurred = cv2.GaussianBlur(cv_image,(11,11),0)
        mask = cv2.inRange(cv_image_hsv,orange_lower,orange_upper)
        mask = cv2.erode(mask,None,iterations=2)
        mask = cv2.dilate(mask,None,iterations=2)
        self.mask = mask

	    # find contours in the mask and initialize the current
	    # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
	    	cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        center = None
	    #print (cnts)
	    # only proceed if at least one contour was found
        if len(cnts)>0:
            c = max(cnts, key=cv2.contourArea)
            ((x,y),radius) = cv2.minEnclosingCircle(c)
            self.ballloc_pixel = [int(x),int(y)]
            if radius > 3:
                cv2.circle(self.cv_image, (int(x),int(y)), int(radius),(0,0,255),2)

    def depth_cb(self,data):
        try:
            self.depth_image = bridge.imgmsg_to_cv2(data,"16UC1")
        except CvBridgeError as e:
            print(e)

if __name__ == "__main__":
    rospy.init_node("measure_3d")
    tracker = Tracker3D()
    viz_img = True
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        tracker.get_xyz()
        tracker.pub_xy()
        tracker.pub_viz()
        print("Ball location: ({},{})".format(tracker.ballloc_xyz[0],tracker.ballloc_xyz[1]))
        if viz_img:
            # tracker.viz_3d()
            cv2.imshow("Image window",tracker.cv_image)
            # cv2.imshow("Image window",tracker.mask)
            cv2.waitKey(1) & 0xFF
        rate.sleep()