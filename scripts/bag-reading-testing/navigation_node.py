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
from geometry_msgs.msg import Vector3, Transform
import std_msgs
from std_msgs.msg import Int8



pub = rospy.Publisher('/locobot/mobile_base/commands/velocity', Twist, queue_size=10)


def nav_callback(msg):
    
    april_tag_msg = msg
    
    x = april_tag_msg.translation.x
    y = april_tag_msg.translation.y
    z = april_tag_msg.translation.z

    # read in position of april tags and convert it to robot frame

    # pos = ???

    twist = Twist() 

    #twist.linear.x = y
    #twist.linear.y = x

    ## This doesn't work yet but idea is to turn towards the april tag and then to drive towards it.

    ## This should be messed with to improve the controller.
    print(twist.linear.x)
    print(twist.linear.y)
    twist.angular.z = -x/2
    rospy.loginfo("publish")
    pub.publish(twist)
    rospy.sleep(1)

    twist = Twist() 
    twist = Twist() 

    twist.linear.x = y

    rospy.loginfo("publish")
    pub.publish(twist)

    rospy.sleep(3)

    #new_pub.publish(Int8(3))

    #sleep(1)

    #twist = Twist()
    



def nav_april_tag_subscriber():
    rospy.init_node('move_robot', anonymous=True)
    rospy.Subscriber("apriltag_camera_coord", Transform, nav_callback)
    rospy.spin()

if __name__ == '__main__':
    
    nav_april_tag_subscriber()

