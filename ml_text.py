#!/usr/bin/env python
import rospy
import cv2
import tensorflow as tf 
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D, BatchNormalization
#import tensorflow_hub as hub
import numpy as np 
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

graph = tf.get_default_graph()

h=192
b=256
colours=['Green','Navy_Blue','Cyan','Red','Yellow','Brown','Black','Grey']

inputs=tf.keras.Input(shape=(h,b,1))

x=Conv2D(16,3,1,padding='same',activation='relu')(inputs)
x=MaxPool2D(2,2,padding='valid')(x)

x=Conv2D(32,3,1,padding='same',activation='relu')(x)
x=MaxPool2D(2,2,padding='valid')(x)

x=Conv2D(32,3,1,padding='same',activation='relu')(x)
x=BatchNormalization()(x)
x=MaxPool2D(2,2,padding='valid')(x)

x=Conv2D(64,3,1,padding='same',activation='relu')(x)
x=MaxPool2D(2,2,padding='valid')(x)

x=Conv2D(64,3,1,padding='same',activation='relu')(x)
x=MaxPool2D(2,2,padding='valid')(x)

x=Flatten()(x)
output=Dense(8,activation='softmax')(x)

model=tf.keras.Model(inputs,output,name='model2a')

model.compile(optimizer='adagrad',
	loss='sparse_categorical_crossentropy',
	metrics=['accuracy'])

PATH='/home/amit/Team-6-ML-subsystem/model1a_9927.h5'
#model.load_weights(PATH)


def callback(ros_image_message):
	bridge = CvBridge()
	img = bridge.imgmsg_to_cv2(ros_image_message, 
	desired_encoding='passthrough')

	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img=cv2.resize(img,(b,h))
	img=img.reshape(1,h,b,1)
	img=img/255.0
	#print(img[0][0][0][0])
	#img_new=cv2.resize(img,(64*10,48*10))
	#cv2.imshow('f',img_new)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#model=tf.keras.models.load_model((PATH))
	global graph
	with graph.as_default():
		pass
		preds=model.predict([img])
		result=np.argmax(preds,axis=1)

	print(result)
	print(preds[0][result])	

def listener():
	rospy.init_node('ml_text',anonymous=True)
	rospy.Subscriber('/hexbot/camera1/image_raw',
		Image,callback)
	rospy.spin()

if __name__=='__main__':
	listener()






