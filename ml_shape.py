import tensorflow as tf 
import numpy as np 
import cv2

h=48
b=64
model=tf.keras.models.load_model('model1a_9927.h5')

img=cv2.imread('cyan.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(b,h))
img=img.reshape(1,h,b,1)
img=img/255.0
preds=model.predict(img)
res=np.argmax(preds,axis=1)
if res==0:
	shape="torus"
else :
	shape="not torus"
print(shape,preds[0][int(res)])
