# import numpy as np

# def mse(imageA,iamgeB):
#     err=np.sum((imageA.astype('float')-iamgeB.astype('float'))**2)
#     err/=float(imageA.shape[0]*iamgeB.shape[1])

#     return err

from skimage.measure import compare_ssim as compare_ssim
from skimage.measure import compare_mse as mse
import matplotlib.pyplot as pyplot
import numpy as np
import cv2

cap=cv2.VideoCapture(0)
print(cap)

frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height=int(480/frame_width*frame_height)

ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)

title='camera compare'
plt.ion()

#cap.read()
ret,frame=cap.read()
temp=cv2,cvtColor(frame,cv2.COLOR_BGR2BGRA)
#TODO 前10帧
while cap.isOpened():
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    m=mse(temp,gray)
    s=ssim(temp,gray)
    print('MSE: %.2f,SSIM:%.2f'%(m,s))

    #show first image
    #ax=fig.add_subplot(1,2,1)
    #plt.imshow(temp,cmap=plt.cm.gray)
    #plt.axis('off')

    #shwo the second image
    #ax=fig.add_subplot(1,2,2)
    #plt.imshow(gray,cmap=plt.cm.gray)
    #plt.axis('off')

    #show the image
    #plt.show