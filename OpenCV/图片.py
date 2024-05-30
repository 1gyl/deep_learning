import numpy as np
import cv2
from matplotlib import pyplot as plt
import os 
import errno
import sys

# print(cv2.__version__)

# #img=cv2.imread('',cv2.IMREAD_COLOR)#读入一幅彩色图像，图像的透明度会被忽略，默认参数
# #img=cv2.imread('',cv2.IMREAD_GRAYSCALE)# Load an color image in grayscale 灰度
# img=cv2.imread('1.jpeg',cv2.IMREAD_UNCHANGED)#包括图像的alpha通道

# img=cv2.resize(img,(640,480))
# rows,cols,ch=img.shape
# print('行/高:',rows,'列/宽:',cols,'通道:',ch)
# #图像的宽对应的是列数，高对应的行数

# cv2.namedWindow('img',cv2.WINDOW_NORMAL)#可调整窗口大小
# #img=cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)#自动调整
# #img=cv2.namedWindow('img',cv2.WINDOW_KEEPRATIO)#保持图片比例

# cv2.imshow('img',img)#图片会自动调整为图像大小
# #在窗口上按任意建退出
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img=cv2.imread('1.jpeg',0)
# plt.imshow(img,cmap='gray',interpolation='bicubic')#imgs是要显示的图像数据，cmap采纳数指定了使用灰度色彩映射(gray)，interpolation参数指定了茶值方法为双三次插值，可以提高图像的平滑度
# #彩色图像使用OpenCV加载时是BGR模式。但是Matplotlib是RGB模式。所以彩色图像如果已经被OpenCV读取，他将不会被Matplotlib正确显示

# plt.xticks([]),plt.yticks([]) #to hide tick values on X and Y axis
# plt.show()

# path='1.jpeg'
# if not os.path.exists(path):
#     raise FileNotFoundError(errno.ENOENT,os.strerror(errno.ENOENT),path)

# img=cv2.imread(path,cv2.IMREAD_UNCHANGED)

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# for i in range(1,len(sys.argv)):
#     print(f'参数{i}:{sys.argv[i]}')

# if len(sys.argv)>2:
#     print('图片.py 1.jpeg')
#     sys.exit(-1)

# image_path=sys.argv[1]
# try:
#     f=open(image_path)
# except Exception as e:
#     print(e)
#     sys.exit(-1)

# img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)#包括图像的alpha通道
# temp=img.copy()

# title=image_path.split('/')[-1]+f'{img.shape}'

# gray=False
# while True:
#     cv2.imshow(title,temp)

#     k=cv2.waitKey(10)
#     if k==27 or k == ord('q'):
#         break
#     #TODO 分辨率太大，需要缩放
#     if k==ord('g'):
#         #t=temp==ing
#         #if t.all():
#         #if t.any():
#         #if temp==img;
#         if gray is False:
#             temp=cv2,cvtColor(img,cv2.COLOR_BGR2BGRA)
#             gray=True
#         else:
#             temp=img.copy()
#             gray=False
# cv2.destroyAllWindows()

# #创建黑白图片
# size=(2560,1600)
# #全黑，可以用在屏保
# black=np.zeros(size)
# print(black[34][56])
# cv2.imwrite('black.jpg',black)

# #全白
# black[:]=255
# print(black[34][56])
# cv2.imwrite('white.jpg',black)

img=cv2.imread('1.jpeg',cv2.IMREAD_COLOR)
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#颜色转换
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow('img',gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

temp=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
cv2.imshow('img',temp)
cv2.waitKey(0)
cv2.destroyAllWindows()