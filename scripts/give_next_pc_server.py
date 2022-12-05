#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_srvs.srv import Empty
from ptcloud_visualization.srv import GiveNextPc

counter = 0

def give_next_pc(req):
    global counter
    counter += 1
    result = counter
    rospy.loginfo("PLY file number " + str(result))
    return result


if __name__ == '__main__':
    rospy.init_node("give_next_pc_server")
    rospy.loginfo("Give next pc server node created")
    service = rospy.Service("/give_next_pc", GiveNextPc, give_next_pc)
    rospy.loginfo("Server has started")

    rospy.spin()
