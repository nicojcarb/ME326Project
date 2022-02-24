import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

from dt_apriltags import Detector

if __name__ == '__main__':
    visualization = True

    b = rosbag.Bag('2022-02-17-15-17-18.bag')
    print(b)

    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    for topic, msg, t in b.read_messages(topics=['/locobot/camera/color/image_raw']):
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

        tags = at_detector.detect(cv_image_gray)
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

        if visualization:
            cv2.imshow('Detected tags', color_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    b.close()
