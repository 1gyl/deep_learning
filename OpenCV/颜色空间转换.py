# '''
# 在OpenCV中有超过150种进行颜色空间转换的方法。但是你以后就会发现我们经常用到的也就两种BGR…GRAY和BGR…HSV
# 我们用的哦啊的函数是cv2.cvtColor(input_image flag)其中flag就是转换的类型
# 对于BGR…GRAY的转换，我们使用的flag就是cv2.COLOR_BGR2GRAY。同样对于BGR…HSV的转换，我们使用的flag就是cv2.COLOR_BGR2HSV
# '''

# import cv2
# from pprint import pprint

# flags=[i for i in dir(cv2) if i.startswith('COLOR_')]
# pprint(flags)

# '''
# 在OpenCV的HSV格式中H 色彩/色度 的取值范围是[0 179]
# S饱和度的取值范围是[0 255]
# V 来嗯度的取值范围是[0 255]
# 但是不同的件使用的值可能不同
# 所以当你拿OpenCV的HSV值与其他件的HSV值对比时，一定记得归一化
# '''

import cv2
import numpy as np

'''
green=np.uint8([0,255,0])
print(green)
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print(hsv_green)

scn(the number of channels of the source),
i.e,self.img.depth(),is neither CV_8U nor CV_32F
所以不能使用[0,255,0]而使用[[[0,255,0]]]
的三层括号应分别对应于cvArray cvMat IPlImage
'''

green=np.uint([[[0,255,0]]])
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)    
print(hsv_green)
