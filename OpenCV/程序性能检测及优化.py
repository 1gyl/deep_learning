import cv2
import numpy as np

# '''
# 使用OpenCV检测程序效率
# '''

# img1=cv2.imread('1.jpeg')

# e1=cv2.getTickCount()

# for i in range(5,49,2):
#     img1=cv2.medianBlur(img1,i)

# e2=cv2.getTickCount()
# t=(e2-e1)/cv2.getTickFrequency()#时钟频率或者每秒钟的时钟数
# print(t)

'''
OpenCV中的默认优化
在编译时优化是默认开启的。因此OpenCV就是优化后的代码
如果你把优化关闭的就智能执行低效的代码
你可以使用函数cv2.useOptimized()来查看优化是否开启了
使用cv2.setUseOptimized()来开启优化
'''

print(cv2.useOptimized())
cv2.setUseOptimized(False)
print(cv2.useOptimized())