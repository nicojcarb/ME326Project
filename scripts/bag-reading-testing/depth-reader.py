import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CompressedImage
import numpy as np
import matplotlib.pyplot as plt

b = rosbag.Bag('today2.bag')
print(b)
i = 0
for topic, msg, t in b.read_messages(topics=['/locobot/camera/depth/image_rect_raw']):
# replace the topic name as per your need
	CAMERA_MSG = msg
	img = Image()
	#camera.display()
	camera = CAMERA_MSG

	img.data = camera.data
	img.height = camera.height
	img.width = camera.width
	img.encoding = camera.encoding
	img.is_bigendian = camera.is_bigendian
	img.step = camera.step	
	bridge = CvBridge()


	cv_image = bridge.imgmsg_to_cv2(img, '32FC1')
	cv2.imshow("hello:", cv_image)
	cv_image_array = np.array(cv_image, dtype=np.uint16)*0.001
	# cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
	# Normalize the depth image to fall between 0 (black) and 1 (white)
	# http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
	cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
	# Resize to the desired size


	cv2.imshow("Image from my node", cv_image_norm)

	cv2.waitKey(0) 
	print(i)
	i += 1



	
b.close()



'''
CAMERA_MSG = b.message_by_topic('locobot/camera/color/image_raw')
camera = pd.read_csv(CAMERA_MSG)

print(camera.__dict__)

from cv_bridge import CvBridge
bridge = CvBridge()
print(type(camera.iloc[0].data))

from sensor_msgs.msg import Image
import cv2

img = Image()
#camera.display()
camera = camera.iloc[0]
print(camera)
img.data = camera.data
img.height = camera.height
img.width = camera.width
img.encoding = camera.encoding
img.is_bigendian = camera.is_bigendian
img.step = camera.step	
#print(type(img.data))
#print('#####')


cv_image = bridge.imgmsg_to_cv2(img, 'rgb8')

'''