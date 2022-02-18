import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image
import numpy as np
import matplotlib.pyplot as plt

b = rosbag.Bag('today2.bag')
print(b)

def filter_image(color_img, color_mask='r'):
    """Returns the black and white image with a color-mask for the specified color (white or the color where the color is, black everywhere else)
    
    Parameters
    ----------
    color_img : np.ndarray
        Raw input image of colored blocks on a table
    color_mask : string
        String indicating which color to draw the mask on; options 'r', 'g','b','y' (red, green, blue, yellow)
    
    Returns
    -------
    mask_img : np.ndarray
        Image with just the selected color shown (either with true color or white mask)
    """
    #Hint: one way would be to define threshold in HSV, leverage that to make a mask?
 
    #raise NotImplementedError
    #Step 1: Convert to HSV space; OpenCV uses - H: 0-179, S: 0-255, V: 0-255
    
    # Below from https://lindevs.com/convert-image-from-rgb-to-hsv-color-space-using-opencv/
    #img = cv2.imread(color_img)
    hsvImg = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    

    #Step 2: prep the mask
    
    # strategy from https://stackoverflow.com/questions/30331944/finding-red-color-in-image-using-python-opencv
    # https://cvexplained.wordpress.com/2020/04/28/color-detection-hsv/#:~:text=The%20HSV%20values%20for%20true,10%20and%20160%20to%20180.
    if color_mask == 'r':
        # lower boundary RED color range values; Hue (0 - 10)
        lower = np.array([0, 100, 20])
        upper = np.array([10, 255, 255])

        # upper boundary RED color range values; Hue (160 - 180)
        lower2 = np.array([160,100,20])
        upper2 = np.array([179,255,255])
        #do
    elif color_mask == 'g':
        lower = np.array([36, 50, 20])
        upper = np.array([86, 255, 255])
        lower2 = lower
        upper2 = upper
        #do
    elif color_mask == 'b':
        lower = np.array([100, 210, 50])
        upper = np.array([135, 255, 200])
        lower2 = lower
        upper2 = upper
        #do
    elif color_mask == 'y':
        lower = np.array([20, 80, 50])
        upper = np.array([34, 255, 255])
        lower2 = lower
        upper2 = upper
        #do
    
    lower_mask = cv2.inRange(hsvImg, lower, upper)
    upper_mask = cv2.inRange(hsvImg, lower2, upper2)
    
    mask = lower_mask + upper_mask
    
    #Step 3: Apply the mask; black region in the mask is 0, so when multiplied with original image removes all non-selected color 
    mask_img = cv2.bitwise_and(color_img, color_img, mask = mask)
    
    return mask_img

def good_feature_tracker(img_input, maxCorners=100, qualityLevel=200,min_distance=10):
    """ This function takes in an image and detects corners
    
    Parameters
    ----------
    img_input : np.ndarray
        Raw input image of environment
    maxCorners : float
        Maximum number of corners to return. If there are more corners than are 
        found, the strongest of them is returned.
    qualityLevel : float
        Parameter characterizing the minimal accepted quality of image corners. 
        The parameter value is multiplied by the best corner quality measure, 
        which is the minimal eigenvalue (see cornerMinEigenVal ) or the Harris 
        function response (see cornerHarris ). The corners with the quality 
        measure less than the product are rejected. For example, if the best 
        corner has the quality measure = 1500, and the qualityLevel=0.01 , 
        then all the corners with the quality measure less than 15 are rejected
    min_distance : int
        Minimum possible Euclidean distance between the returned corners
        
    Returns
    -------
    Canny_edges : np.ndarray
        Gray scale image with canny edges
    """
    #Hint: Documentation exists for the cv2 Shi-Tomasi Corner Detector & Good Features to Track...
    #Hint: copying an image: newImage = myImage.copy()
    
    # Following https://docs.opencv.org/3.4/d4/d8c/tutorial_py_shi_tomasi.html
    
    #Step 1: Gray scale the original image
    gray = cv2.cvtColor(img_input,cv2.COLOR_BGR2GRAY)
    
    #Step 2: Extract corners from the image
    corners = cv2.goodFeaturesToTrack(gray,maxCorners,qualityLevel,min_distance)
    corners = np.int0(corners)

    #Step 3: redraw them (as circles) on a copy of the image to return
    img_copy = img_input.copy()
    for i in corners:
        x,y = i.ravel()
        cv2.circle(img_copy,(x,y),3,255,-1)
    return img_copy, corners
#K: [589.3671569219632, 0.0, 320.5, 0.0, 589.3671569219632, 240.5, 0.0, 0.0, 1.0]

for topic, msg, t in b.read_messages(topics=['/locobot/camera/color/image_raw']):
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

	cv_image_raw = bridge.imgmsg_to_cv2(img, 'bgr8')

	'''
	#cv_image = bridge.imgmsg_to_cv2(img, '32FC1')
	cv_image_array = np.array(cv_image, dtype = np.dtype('f8'))
	# Normalize the depth image to fall between 0 (black) and 1 (white)
	# http://docs.ros.org/electric/api/rosbag_video/html/bag__to__video_8cpp_source.html lines 95-125
	cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)
	  # Resize to the desired size
	  '''
	cv_image = filter_image(cv_image_raw, 'b')

	good_features_img, corners = good_feature_tracker(cv_image, maxCorners=16, qualityLevel=0.01,min_distance=10)

	#print(corners)
	#display img
	plt.figure()
	plt.subplot(211)
	plt.imshow(cv_image_raw)
	plt.subplot(212)
	plt.imshow(good_features_img)
	plt.show()

	#cv2.imshow("Image from my node", cv_image)

	#cv2.waitKey(0) 



	
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