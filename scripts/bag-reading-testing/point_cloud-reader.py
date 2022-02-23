import rosbag
import sensor_msgs
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
import numpy as np
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pc2
import ros_numpy
import pptk
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D


b = rosbag.Bag('today2.bag')
print(b)
i = 0


for topic, msg, t in b.read_messages(topics=['/locobot/pc_filter/pointcloud/objects']):

	fig = plt.figure(figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')
# replace the topic name as per your need
	MSG = msg
	point_cloud = PointCloud2()
	#camera.display()
	sensor = MSG

	point_cloud.data = sensor.data
	point_cloud.height = sensor.height
	point_cloud.width = sensor.width
	point_cloud.row_step = sensor.row_step
	point_cloud.point_step = sensor.point_step
	point_cloud.is_dense = sensor.is_dense
	point_cloud.is_bigendian = sensor.is_bigendian

	bridge = CvBridge()


	xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(MSG)

	np.save('xyz_array', xyz_array)
	'''
	v = pptk.viewer(xyz_array)
	v.attributes(colors/65535)
	v.color_map('cool')
	v.set(point_size=0.001,bg_color=[0,0,0,0],show_axis=0,show_grid=0)
	'''
	#print(ax)
	
	def planeFit(points):
	    """
	    p, n = planeFit(points)

	    Given an array, points, of shape (d,...)
	    representing points in d-dimensional space,
	    fit an d-dimensional plane to the points.
	    Return a point, p, on the plane (the point-cloud centroid),
	    and the normal, n.
	    """
	    import numpy as np
	    from numpy.linalg import svd
	    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
	    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
	    ctr = points.mean(axis=1)
	    x = points - ctr[:,np.newaxis]
	    M = np.dot(x, x.T) # Could also use np.cov(x) here.
	    return ctr, svd(M)[0][:,-1]
	    '''
	print(planeFit(xyz_array.T))  
	point, normal = planeFit(xyz_array.T)
	new_xyz = []

	for xyz in xyz_array:
		x = xyz[0]
		y = xyz[1]
		z = xyz[2]
		xp, yp, zp = point[0], point[1], point[2]
		nx, ny, nz = normal[0], normal[1], normal[2]
		distance = (np.abs(nx * (xp - x) + ny * (yp - y) + nz * (zp - z)) / np.sqrt(nx**2 + ny**2 + nz**2))
		if distance > 2e-8:
			new_xyz.append(xyz)
	xyz_array = np.array(new_xyz)
	'''
	kmeans = KMeans(n_clusters=9, random_state=4004).fit(xyz_array)
	cluster_centers = kmeans.cluster_centers_
	xyz_array = cluster_centers

	proj_of_u_on_v = [(xyz - (np.dot(xyz, normal)/np.linalg.norm(normal)**2)*normal) for xyz in xyz_array]

	xyz_array = np.array(proj_of_u_on_v)
	'''
	'''

	d = -point.dot(normal)
	
	# create x,y
	xx, yy = np.meshgrid(np.linspace(-.4,.2, 1000), np.linspace(-.4,.2, 1000))

	# calculate corresponding z
	z = (-normal[0] * xx - normal[1] * yy - d) * 1. /normal[2]
	
	#plt3d = fig.gca(projection='3d')
	#plt3d.plot_surface(xx, yy, z)

	print(xyz_array)
	ax = plt.gca()
	
	ax.scatter(xyz_array[:, 0], xyz_array[:, 1], xyz_array[:, 2], color = 'r')
	# plot the surface

	plt.show()


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