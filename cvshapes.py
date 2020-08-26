import cv2
import numpy as np 

#img=cv2.imread()

def load(img_name):
  img=cv2.imread(img_name,1)
  img=cv2.resize(img,(960,540))
  img_new=img[106:488,245:780]
  #img_new=cv2.resize(img_new,(b,h))
  #img_new=img_new.reshape(h,b,1)
  return img_new

#if using solid work images and want to crop out some stuff
img=load('cyan.png')

#Otherwise
#img=cv2.imread('b.png')

img_rgb=img.copy()
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_,thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY_INV)
_, contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,
	cv2.CHAIN_APPROX_SIMPLE)
#print(len(contours))
area=0
for cnts in contours:
	if cv2.contourArea(cnts)>area:
		cnt=cnts
		area=cv2.contourArea(cnts)

peri=cv2.arcLength(cnt,True)
epsilon=0.01*peri
approx=cv2.approxPolyDP(cnt,epsilon,True)
img=cv2.drawContours(img,[cnt],0,(0,255,0),3)

x,y,w,h=cv2.boundingRect(cnt)

_,(wr,hr),_=cv2.minAreaRect(cnt)

ellipse=cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(0,0,255),2)
a=ellipse[1][0]
b=ellipse[1][1]

if len(approx)<=6 :
	shape="cuboid"

elif 0.95<w*1.0/h<1.05 :
	shape="sphere"
elif wr*hr>w*h+0.1*area :
	shape="cone"
elif w*1.0/h<1.5 :
	shape="it's tall not wide"
elif abs(0.786*a*b-area)<0.05*area:
	shape="torus"
else :
	shape="I don't know"

mask=np.zeros_like(gray)
mask=cv2.drawContours(mask,[cnt],0,255,-1)
cv2.imshow('mask',mask)

mask=mask/255.0
num=np.sum(mask)

r=np.sum(img_rgb[:,:,2]*mask)/num
g=np.sum(img_rgb[:,:,1]*mask)/num
b=np.sum(img_rgb[:,:,0]*mask)/num

color="unidentified"

if (r-g)**2+(g-b)**2+(b-r)**2<100 :
	bright=np.sum(gray*mask)/num
	if bright<10 :
		color="black"
	else :
		color="gray"

else :
	img_hsv=img_hsv*1.0
	hue=np.sum(mask*img_hsv[:,:,0])/np.sum(mask)
	if hue<=5 or hue>=175 :
		color="red"
	else :
		colors=["green","navy_blue","cyan","yellow","brown"]
		hue_values=[60,120,90,30,15]
		for i in range(5) :
			if hue_values[i]-5<=hue<=hue_values[i]+5 :
				color=colors[i]

print(shape)
print(color)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



