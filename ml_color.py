import tensorflow as tf 
import numpy as np 
import cv2

colours=['Green','Navy_Blue','Cyan','Red','Yellow','Brown','Black','Grey']
h=48
b=64
model=tf.keras.models.load_model('model1b_10000.h5')

img=cv2.imread('cyan.png')
img=cv2.resize(img,(b,h))
img=img.reshape(1,h,b,3)
img=img/255.0
preds=model.predict(img)
res=np.argmax(preds,axis=1)
print(colours[int(res)],preds[0][int(res)])
