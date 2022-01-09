#!/usr/bin/env python3
import os
import rospy
from std_msgs.msg import String
import torch
import transformers

device = None
path = os.getcwd()
model_path = path + '/umrf_t5_small_fine_tuned'
tokenizer_path = path + '/saved_tokenizer'

model = transformers.T5ForConditionalGeneration.from_pretrained(model_path)
model.eval()
tokenizer = transformers.T5Tokenizer.from_pretrained(tokenizer_path)

pub = rospy.Publisher('umrf_parses_output', String, queue_size=10)

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
    encoded_input = tokenizer(data.data, max_length=512, padding='longest', truncation=True, return_tensors='pt').data
    
    input_ids = encoded_input['input_ids'].to(device)
    print(input_ids)

    umrf_parse = model.generate(input_ids, max_length=1000)
    
    print(umrf_parse[0])
    pred = tokenizer.decode(umrf_parse[0])
    
    pub.publish(pred)



def listener():
    rospy.init_node('umrf_parser_node', anonymous=True)

    rospy.Subscriber("umrf_parses", String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('gpu found')
    else:
        device = torch.device('cpu')
        print('using cpu')
    try:
        listener()
    except rospy.ROSInterruptException:
        pass
