import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image

b = rosbag.Bag('testing.bag')

for topic, msg, t in b.read_messages(topics=['locobot/camera/color/image_raw']):
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

	cv_image = bridge.imgmsg_to_cv2(img, 'rgb8')

	cv2.imshow('image', cv_image)

	cv2.waitKey(0) 

	

	
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