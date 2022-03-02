Place any custom launch files here!

Launch files allow robot to be spawned in custom world with colorful blocks.

Paths for launch files included here:
   
/usr/share/gazebo-11/worlds/project.world 

/opt/ros/noetic/share/gazebo_ros/launch/project_world.launch

~/interbotix_ws/src/interbotix_ros_rovers/interbotix_ros_xslocobots/interbotix_xslocobot_moveit/launch/testbot_moveit.launch

~/interbotix_ws/src/interbotix_ros_rovers/interbotix_ros_xslocobots/interbotix_xslocobot_gazebo/launch/testlaunch.launch

Command to launch robot in block/project world:
roslaunch interbotix_xslocobot_moveit testbot_moveit.launch robot_model:=locobot_wx250s show_lidar:=true use_gazebo:=true dof:=6 use_moveit_rviz:=true
