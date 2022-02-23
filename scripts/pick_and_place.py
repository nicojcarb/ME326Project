import sys
import os
os.environ["ROS_NAMESPACE"] = "/locobot"
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from geometry_msgs.msg import Quaternion
from tf.transformations import quaternion_from_euler

moveit_commander.roscpp_initialize(sys.argv)
rospy.init_node('locobot', anonymous=True)

robot = moveit_commander.RobotCommander("robot_description")
scene = moveit_commander.PlanningSceneInterface()    
print(robot.get_group_names())
group = moveit_commander.MoveGroupCommander("interbotix_arm")
display_trajectory_publisher = rospy.Publisher('/move_group/display_planned_path', moveit_msgs.msg.DisplayTrajectory, queue_size = 10)

group_gripper = moveit_commander.MoveGroupCommander("interbotix_gripper")

group_variable_values = group.get_current_joint_values()

print(group_gripper.get_active_joints())

print(group_gripper.get_current_joint_values())

print(group_variable_values)

'''
group_variable_values[0] = 1 ## Change these joint commands (in radians)
group_variable_values[1] = 1
group_variable_values[3] = -1
group_variable_values[5] = 1.5
group.set_joint_value_target(group_variable_values)
'''
pose_goal = geometry_msgs.msg.Pose()
pose_goal.orientation.w = 1.0
pose_goal.position.x = 0.4
pose_goal.position.y = 0.1
pose_goal.position.z = 0.4

grasp = moveit_msgs.msg.Grasp()
grasp.grasp_pose.header.frame_id = robot.get_planning_frame()
q = quaternion_from_euler(0, 0, 3.14/2)
orientation = Quaternion(q[0], q[1], q[2], q[3])
grasp.grasp_pose.pose.orientation = orientation
print("orientation:")
print(grasp.grasp_pose.pose.orientation)
print(type(grasp.grasp_pose.pose.orientation))
x = geometry_msgs.msg.Point(0.5, 0, .5)
grasp.grasp_pose.pose.position = x
print("position:")
print(grasp.grasp_pose.pose.position)
print(type(grasp.grasp_pose.pose.position))
grasp.pre_grasp_approach.direction.header.frame_id = robot.get_planning_frame()
#grasp.pre_grasp_approach.direction.vector.x = 1.0
#grasp.pre_grasp_approach.min_distance = 0.095

def openGripper(posture):
	print(posture)
	posture.joint_names = [0, 0]
	posture.joint_names[0] = "locobot/left_finger_link"
	posture.joint_names[1] = "locobot/right_finger_link"
	posture.points = []
	posture.points[0].positions[0] = 0.04
	posture.points[0].positions[1] = 0.04
	posture.points[0].time_from_start = rospy.Duration(0.5, 0)

def closedGripper(posture):
	posture.joint_names[0] = "locobot/left_finger_link"
	posture.joint_names[1] = "locobot/right_finger_link"
	posture.points[0].positions[0] = 0
	posture.points[0].positions[1] = 0
	posture.points[0].time_from_start = rospy.Duration(0.5, 0)

#openGripper(grasp.pre_grasp_posture);
scene.remove_world_object("pole")
p = geometry_msgs.msg.PoseStamped()
p.header.frame_id = robot.get_planning_frame()
p.pose.position.x = 0.7
p.pose.position.y = -0.4
p.pose.position.z = 0.85
p.pose.orientation.w = 1
scene.add_box("pole", p, (0,3, 0.1, 1.0))
print(group.__dict__)
group.pick("pole", grasp)
group_gripper_values = group_gripper.get_current_joint_values()
group_gripper.set_joint_value_target(group_variable_values)

print(robot.gripper)

group.set_pose_target(pose_goal)


plan2 = group.plan()
group.go(wait=True)

rospy.sleep(5)

group.stop()

moveit_commander.roscpp_shutdown()