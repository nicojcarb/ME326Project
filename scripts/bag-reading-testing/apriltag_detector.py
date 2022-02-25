#!/usr/bin/env python3

# from matplotlib.transforms import Transform
import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

import rospy
from dt_apriltags import Detector
import geometry_msgs.msg
from geometry_msgs.msg import Vector3, Transform


def apriltag_callback(msg):
    visualization = False

    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    # replace the topic name as per your need
    CAMERA_MSG = msg
    img = Image()
    camera = CAMERA_MSG

    img.data = camera.data
    img.height = camera.height
    img.width = camera.width
    img.encoding = camera.encoding
    img.is_bigendian = camera.is_bigendian
    img.step = camera.step	
    bridge = CvBridge()

    cv_image_raw = bridge.imgmsg_to_cv2(img, 'bgr8')
    cv_image_gray = cv2.cvtColor(cv_image_raw, cv2.COLOR_RGB2GRAY)

    # camera matrix
    K = [609.7716674804688, 0.0, 331.6385192871094, 0.0, 610.3194580078125, 247.25904846191406, 0.0, 0.0, 1.0]

    cameraMatrix = np.array(K).reshape((3,3))
    camera_params = (cameraMatrix[0,0], cameraMatrix[1,1], cameraMatrix[0,2], cameraMatrix[1,2])

    tags = at_detector.detect(cv_image_gray, True, camera_params, 0.08)
    tag_ids = [tag.tag_id for tag in tags]
    print(len(tags), " tags found: ", tag_ids)

    color_img = cv2.cvtColor(cv_image_gray, cv2.COLOR_GRAY2RGB)

    for tag in tags:
        for idx in range(len(tag.corners)):
            cv2.line(color_img, tuple(tag.corners[idx-1, :].astype(int)), tuple(tag.corners[idx, :].astype(int)), (0, 255, 0))

        cv2.putText(color_img, str(tag.tag_id),
                    org=(tag.corners[0, 0].astype(int)+10,tag.corners[0, 1].astype(int)+10),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.8,
                    color=(0, 0, 255))

    # publish coordinates
    # print(tags[0])
    at_coord_publisher(tags[0].pose_t)

    if visualization:
        cv2.imshow('Detected tags', color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def raw_img_subscriber():
    rospy.init_node('raw_img_subscriber', anonymous=True)
    rospy.Subscriber("/locobot/camera/color/image_raw", Image, apriltag_callback)
    rospy.spin()

def at_coord_publisher(t): # TODO: add 
    print(t)
    pub = rospy.Publisher('apriltag_camera_coord', Transform, queue_size=10)
    #rospy.init_node('apriltag_coord', anonymous=True)
    rate = rospy.Rate(0.75)
    
    # while not rospy.is_shutdown():
    my_message = Transform()
    vector = Vector3()
    vector.x = t[0]
    vector.y = t[1]
    vector.z = t[2]
    my_message.translation = vector
    pub.publish(my_message)
    rate.sleep()

if __name__ == '__main__':
    raw_img_subscriber()
