import cv2
import numpy as np
from matplotlib import pyplot as plt

# #为图像扩边，填充
# #如果你想再图像周围创建一个边框，就像相框一样
# #经常在卷积运算或0填充时被用到

# BLUE=[255,0,0]

# img1=cv2.imread('1.jpeg')

# replicate=cv2.copyMakeBorder(img1,top=10,bottom=10,left=10,right=10,borderType=cv2.BORDER_REPLICATE)

# reflect=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
# reflect101=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
# wrap=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)

# #constant=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)#value边界颜色

# plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
# plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
# plt.subplot(233),plt.imshow(reflect,'gray').plt.title('REFLECT')
# plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('WRAP')
# plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
# #plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
# plt.show()

# img=cv2.imread("1.jpeg")

# cv2.imshow("1.1.jpeg",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# ball=img[280:340,940:1000]
# img[273:333,100:160]=ball #修改像素值

# cv2.namedWindow("messi",0)
# cv2.imshow("messi",img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# img=cv2.imread('1.jpeg')

# #
# px=img[100,100]
# print(px)
# blue=img[100,100,0]
# print(blue)

# #img[100,100]=[255,255,255]
# print(img[100,100])

# #获取像素值及修改的更好方法
# print(img.item(10,10,2))
# img.itemset((10,10,2),100)
# print(img.item(10,10,2))

# img=cv2.imread('1.jpeg',0)
# print(img.shape)

# img=cv2.imread('1.jpeg')
# rows,cols,ch=img.shape
# print('行/高:',rows,'列/宽:',cols,'通道:',ch)

# print(img.size)
# print(img.dtype)#uint8
# #注意，在debug时，img.dtype非常重要。因为在OpenCV-Python代码中经常出现数据类型的不一致

#拆分及合并图像通道
# img=cv2.imread('1.jpeg')

# b,g,r=cv2.split(img)#比较耗时的操作，请使用numpy索引
# img=cv2.merge((b,g,r))

# #
# b=img[:,:,0]

# #使所有像素的红色通道值都为0，你不必线差分再赋值
# #你可以直接使用Numpy索引，这样会更快
# img[:,:,2]=0

# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

