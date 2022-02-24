import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt
import geometry_msgs.msg 
from geometry_msgs.msg import Twist

import rospy
import time
from time import sleep



pub = rospy.Publisher('/locobot/mobile_base/commands/velocity', Twist, queue_size=10)


def nav_callback(msg):
    april_tag_msg = msg
    

    # read in position of april tags and convert it to robot frame

    # pos = ???

    twist = Twist() 

    twist.linear.x = pos.x
    twist.linear.y = pos.y

    pub.publish(twist)

    sleep(1)

    twist = Twist()
    pub.publish(twist)



def nav_april_tag_subscriber():
    rospy.init_node('move_robot', anonymous=True)
    rospy.Subscriber("apriltag_coord_publisher", Transform, nav_callback)
    rospy.spin()

if __name__ == '__main__':
    nav_april_tag_subscriber()
