import sys
import os
os.environ["ROS_NAMESPACE"] = "/locobot"
import copy
import rospy
import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg

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

group_gripper_values = group_gripper.get_current_joint_values()
#group_gripper.set_joint_value_target(group_variable_values)
group_gripper.go([0.035, -.035], wait = True)

print(robot.gripper)

group.set_pose_target(pose_goal)


plan2 = group.plan()
group.go(wait=True)

rospy.sleep(5)

group.stop()

moveit_commander.roscpp_shutdown()