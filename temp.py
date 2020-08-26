import tensorflow as tf 
import numpy as np 
import cv2

h=48
b=64
model=tf.keras.models.load_model('model1a_9927.h5')
'''
def load(img_name):
  img=cv2.imread(img_name,0)
  img=cv2.resize(img,(960,540))
  img_new=img[106:488,245:754]
  img_new=cv2.resize(img_new,(b,h))
  img_new=img_new.reshape(h,b,1)
  return img_new
'''
#img=load('Screenshot (9).png')
img=cv2.imread('b.png')
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img=cv2.resize(img,(b,h))
img=img.reshape(1,h,b,1)
img=img/255.0
preds=model.predict(img)
print(preds)
img=(255*img).reshape(b,h)
#cv2.imshow('frame',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
