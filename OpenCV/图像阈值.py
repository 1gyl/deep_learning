# import cv2
# import numpy as np

# img=cv2.imread('1.jpeg',0)
# blur=cv2.GaussianBlur(img,(5,5),0)
# #find normalized_histogram,and its cumulative distribution function
# #算归一化直方图
# #CalcHist(image,accumulate=0,mask=NULL)

# hist=cv2.calcHist([blur],[0],None,[256],[0,256])
# hist_norm=hist.ravel()/hist.max()
# Q=hist_norm.cumsum()

# bins=np.arange(256)
# fn_min=np.inf
# thresh=-1

# for i in range(1,256):
#     p1,p2=np.hsplit(hist_norm,[i])#probabilities
#     q1,q2=Q[i],Q[255]-Q[i] #cum sum of classes
#     b1,b2=np.hsplit(bins,[i])# weights

#     #finding means and variances
#     m1,m2=np.sum(p1*b1)/q1,np.sum(p2*b2)/p2
#     v1,v2=np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

#     #calculates the minimization function
#     fn=v1*q1+v2*q2
#     if fn<fn_min:
#         fn_min=fn
#         thresh=i
    
# #find otsu's threshold value with OpenCV function
# ret,otsu=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# print(thresh,ret)

# '''
# 这里用到的函数是cv2.threshold()但是需要多传入一个参数flag cv2.THRESHOLD_OTSU
# 这是把值为0.然后算法会找到最优阈值，这个最优值就是回值retVal.
# 如果不使用Otsu二值化返回的retVal值与设定的阈值相等

# 下面的例子中，输入图像是一幅带有噪声的图像，
# 第一种方法 我们设127为全局阈值
# 第二种方法我们直接使用Otsu二值化
# 第三种方法我们使用一个5*5的高斯去燥，然后再使用Otsu二值化
# 看看噪声，去除对结果的影响有多达吧
# '''
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# img=cv2.imread('1.jpeg')
# #global thresholding
# ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
# #Otsu's thresholding
# ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# #Otsu's thresholding after Gaussian filtering
# #5,5为高斯核的到小为0的标准差
# blur=cv2.GaussianBlur(img,(5,5),0)
# #阈值一定为0
# ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# #plot all the images and there histograms
# images=[img,0,th1,
#         img,0,th2,
#         blur,0,th3]
# titles=['Original Noisy Image','Histogram','Global Thresholding(v=127)',
#         'Original Noisy Image','Histogram',"Otsu's Thresholding",
#         'Gaussian filtered Image','Histogram',"Otsu's Thresholding"
#         ]
# #使用了pyplot中画直方图的方法plt.hist,
# #注意的是它的参数是一维数组
# #所以使用了numpy ravel方法，将多为数组换成一维也可以使用flatten方法
# #ndarray.flat 1-D iterator over an array
# #ndarray.flatten 1-D array copy of the elements of an array in row-major order

# for i in range(3):
#     plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
#     plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
#     plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
#     plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
#     plt.subplot(3,3,i*3+3),plt.imshow(iamges[i*3+2],'gray')
#     plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
# plt.show()
# '''
# 自适应阈值

# Adaptive Method -指定算阈值的方法
# -cv2.ADAPTIVE_THRESH_MEAN_C 值取自相邻区域的平均值
# -cv2.ADAPTIVE_GAUSSIAN_C 值去只相邻区域的加权和，权重为一个高斯窗口
# '''
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# #img=cv2.imread('dave.jpg',0)
# img=cv2.medianBlur(img,5)
# ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# #11为Block size邻域大小用来计算阈值的区域大小
# #2为C值，常数，阈值就等与的平均值或者加权平均值减去这个常数
# th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,11,2)
# th3=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

# titles=['Original Image','Global Thresholding(v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']
# images=[img,th1,th2,th3]

# for i in range(4):
#     plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])




# import cv2
# import numpy as np
# from matplotlib import pyplot as plt

# #img=cv2.imread('dave.jpg',0)
# img=cv2.imread('1.jpeg',0)
# #中值滤波
# img=cv2.medianBlur(img,5)
# ret,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

# #11为Block size邻域大小，
# #2为C值，常数，阈值就等与的平均值或者加权平均值减去这个常数
# th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

# titles=['Original Image','Global Thresholding(v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding ']

# images=[img,th1,th2,th3]

# for i in range(4):
#     plt.subplot(2,2,i+2),plt.imshow(images[i],'gray')
#     plt.title(titles[i])
#     plt.xticks([]),plt.yticks([])

# plt.show()

import cv2
import numpy as np
from matplotlib import pyplot as plt


img=cv2.imread("/home/gyl/OpenCV/1.jpeg")
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#Global thresholding
ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#Otsu's thresholding
ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


#Otsu's thresholding after Gaussian filtering
#5,5为高斯核的大小 0为标准差
blur=cv2.GaussianBlur(img,(5,5),0)
#阈值一定为0
ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plot all the iamges and their histograms
images=[img,0,th1,img,0,th2,blur,0,th3]
titles=['Original Noisy Image','Histogram','Global Threshold(v=127)','Original Noisy Image','Histogram',"Otsu's Thresholding",'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]
#使用了pyplot中画值房图的方法plt.hist,
#注意的是它的参数是一维数组
#所以使用了numpy ravel方法将多维数组换成一维也可以是哟机构年flatten方法
#ndarray.flat 1-D iterator over an array
#ndarray.flatten 1-D array copy of the elements of an array in row-major order

for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.hist(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]),plt.xtick([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
plt.show()