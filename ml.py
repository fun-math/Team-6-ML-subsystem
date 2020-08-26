import tensorflow as tf
import numpy as np 
import cv2

model=tf.keras.models.load_model('model1b_10000.h5')

cap=cv2.VideoCapture('amit.mp4')
h=48
b=64
colours=['Green','Navy_Blue','Cyan','Red','Yellow','Brown','Black','Grey']

while True :
	ret,frame=cap.read()
	if not ret :
		break
	frame=cv2.resize(frame,(960,540))
	cv2.imshow('frame',frame)

	frame_crop=frame[175:325,150:350]
	cv2.imshow('crop',frame_crop)
	#img=cv2.cvtColor(frame_crop,cv2.COLOR_BGR2GRAY)
	img=cv2.resize(frame_crop,(b,h))
	img=img.reshape(1,h,b,3)
	img=img/255.0

	preds=model.predict(img)
	res=np.argmax(preds,axis=1)
	print(colours[int(res)],preds[0][res])
	#print(frame.shape)

	if cv2.waitKey(1) & 0xFF==ord('q'):
		break

cap.release()
