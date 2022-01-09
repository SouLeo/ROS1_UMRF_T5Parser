#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import String
import torch
import transformers

parsing_requests = []
pub = rospy.Publisher('umrf_parses', String, queue_size=10)

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    parsing_requests.append(data.data)


def listener(model):
    rospy.init_node('umrf_parser_node', anonymous=True)

    rospy.Subscriber("umrf_parses", String, callback)


    if parsing_requests:
        umrf_parse = model.generate(parsing_requests.pop(0), max_length=1000)
        pub.publish(umrf_parse)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    local_path = '/home/selmawanna/PycharmProjects/T5LM/saved_model'
    model = transformers.AutoModel.from_pretrained(local_path)
    try:
        listener(model)
    except rospy.ROSInterruptException:
        pass

