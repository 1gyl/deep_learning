import numpy as np
import cv2

img=cv2.imread('R-C.jpeg',cv2.IMREAD_COLOR)
print(img.shape)
logo=cv2.imread('1.jpeg',cv2.IMREAD_COLOR)
print(logo.shape)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# #read color values at position y,x
# y=100
# x=50
# (b,g,r)=img[y,x]
# #print color values to screen
# print('bgrL:',b,g,r)

#先行后列
# logo[100:100+img.shape[0],300:300+img.shape[1]]=img[:,:,0:3]#两张图片的shape不一样
# cv2.imshow('logo',logo)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

font=cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img,text='名侦探柯南',org=(0,0),fontFace=font,fontScale=15,color=(0,255,0),thickness=2,bottomLeftOrigin=True) #text
cv2.putText(img,text='col=width=X10,row=height-Y30',org=(100,300),fontFace=font,fontScale=0.5,color=(255,0,0),thickness=2)#text
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('img+logo',img)
cv2.imwrite('img_logo.jpg',img)
cv2.moveWindow('img+logo',x=img.shape[0],y=0)
cv2,waitKey(0)

