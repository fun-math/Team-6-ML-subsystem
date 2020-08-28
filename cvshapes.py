import cv2
import numpy as np 

#img=cv2.imread()

def load(img_name):
  img=cv2.imread(img_name,1)
  img=cv2.resize(img,(960,540))
  img_new=img[106:488,245:780]
  return img_new

#if using solid work images and want to crop out some stuff
cv_image=load('cyan.png')

#Otherwise
#cv_image=cv2.imread('b.png')

gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY_INV)
_, contours, hierarchy= cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


ind=0
dist=800
for i in range(len(contours)) :
	M = cv2.moments(contours[i])
	cx = int(M['m10']/M['m00'])
	cy = int(M['m01']/M['m00'])
	b=cv_image.shape[1]//2
	if abs(b-cx)<dist :
		dist=abs(b-cx)
		ind=i

cnt=contours[ind]
x,y,w,h=cv2.boundingRect(cnt)
cx=x+w//2
cy=y+h//2
img=cv_image[int(max(cy-1.2*h//2,0)):int(min(cy+1.2*h//2,800)),int(max(cx-1.2*w//2,0)):int(min(cx+1.2*w//2,800))]

#img is the processed image which is a closely cropped image of the object at the centre with a padding of 0.4 times dimensions.

img_rgb=img.copy()
img_hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_,thresh = cv2.threshold(gray,230,255,cv2.THRESH_BINARY_INV)
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
elif abs(0.786*a*b-area)<0.1*area:
	shape="torus"
else :
	shape="cylinder"

mask=np.zeros_like(gray)
mask=cv2.drawContours(mask,[cnt],0,255,-1)
cv2.imshow('mask',mask)

mask=mask/255.0
num=np.sum(mask)

#average value of r,g,b over the contour
r=np.sum(img_rgb[:,:,2]*mask)/num
g=np.sum(img_rgb[:,:,1]*mask)/num
b=np.sum(img_rgb[:,:,0]*mask)/num

color="unidentified"

if (r-g)**2+(g-b)**2+(b-r)**2<100 :
	# if r,g,b are close enough, the colour is some shade of gray  
	bright=np.sum(gray*mask)/num
	if bright<20 :
		color="black"
	else :
		color="gray"

else :
	img_hsv=img_hsv*1.0
	# average hue value over the contour 
	hue=np.sum(mask*img_hsv[:,:,0])/np.sum(mask)
	if hue<=7 :
		color="red"
	elif hue<=23 :
		color="brown"
	elif hue<=45:
		color="yellow"
	elif hue<=75:
		color="green"
	elif hue<=105:
		color="cyan"
	elif hue<=150:
		color="navy_blue"
	else :
		color="red"

print(shape)
print(color)
cv2.imshow('frame',img)
cv2.waitKey(0)
cv2.destroyAllWindows()



