import cv2
import numpy as np
import matplotlib.pyplot as plt

# #学习图像上的运算 加法 减法 位运算等

# #你可以使用函数cv2.add()将两幅图像进行加法运算，当然也可以直接使用numpy
# #res=img1+img
# #两幅图像的大小类型必须一致，或者第二个图像可以使用一个简单的标量值

# x=np.uint8([250])
# y=np.uint8([10])
# print(cv2.add(x,y))
# print(x+y)

#图像混合
img1=cv2.imread('1.jpeg')
print(img1.shape)
img2=cv2.imread('R-C.jpeg')
print(img2.shape)
img1=cv2.resize(img1,(img2.shape[1],img2.shape[0]))

dst=cv2.addWeighted(img1,0.7,img2,0.3,0)#第一幅图的权重是0.7，第二幅图的权重是0.3

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# #按位运算
# #Load two images
# img1=cv2.imread('1.jpeg')
# print(img1.shape)
# img2=cv2.imread('R-C.jpeg')
# print(img2.shape)

# #I want to put R-C on top-left corner,So I create a ROI
# rows,cols,channels=img2.shape
# roi=img1[0:rows,0:cols]

# #Now create a mask of logo and create its inverse mask also
# img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# ret,mask=cv2.threshold(img2gray,10,255,cv2.THRESH_BINARY)
# mask_inv=cv2.bitwise_not(mask)

# #Now black-out the area of logo in ROI
# img1_bg=cv2.bitwise_and(roi,roi,mask=mask_inv)

# #Take only region of logo from logo image
# img2_fg=cv2.bitwise_and(img2,img2,mask=mask)

# #Put logo in ROI and modify the main image
# dst=cv2.add(img1_bg,img2_fg)
# img1[0:rows,0:cols]=dst

# cv2.imshow('res',img1)
# cv2.waitKey(0)
# cv2,destroyAllWindows()

# #图像相减

# img1=cv2.imread('1.jpeg')
# img2=cv2.imread('R-C.jpeg')
# img1=cv2.resize(img1,(img2.shape[1],img2.shape[0]))

# cv2.imshow('subtract1',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow('subtract2',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# st=img2-img1
# #st=img1-img2#相反
# cv2.imshow('after subtract',st)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# st1=img1-img2
# cv2.imshow('after subtract',st1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #效果好一点
# #ret,threshold=cv2.threshold(st,0,127,cv2.THRESH_BINARY)
# ret,threshold=cv2.threshold(st,50,127,cv2.THRESH_BINARY)
# cv2.imshow('after threshold',threshold)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#图像相减2

# #img1=cv2.imread('subtract1.jpg')
# img1=cv2.imread('1.jpeg',0)#灰度图
# #img2=cv2.imread('subtract2.jpg')
# #img2=cv2.imread('subtract2.jpg',0)
# img22=cv2.imread('R-C.jpeg')
# img2=cv2.cvtColor(img22,cv2.COLOR_BGR2GRAY)
# img1=cv2.resize(img1,(img22.shape[1],img22.shape[0]))



# #cv2.imshow('subtract1',img1)
# #cv2.imshow('subtract2',img2)

# st=cv2.subtract(img2,img1)
# st[st<=5]=0 #把小于20的像素点设为0

# #cv2.imshow('after subtract',st)

# '''
# 直方图，看看大部分像素集中在哪个区域
# plt.plot(st)
# pxs=st.ravel()
# pxs=[x for x in pxs if x>5]
# plt.hist(pxs,256,[0,256])
# plt.show()
# '''

# #效果好一点
# #ret,threshold=cv2.threshold(st,0,127,cv2.THRESH_BINARY)
# ret,threshold=cv2.threshold(st,50,255,cv2.THRESH_BINARY)
# #cv2.imshow('after threshold',threshold)

# image,contours=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# areas=()
# for i,cnt in enumerate(contours):
#     # 检查轮廓数据
#     if len(cnt) == 0:
#         print("Error: No contour points found.")
#     else:
#         # 检查点的数据类型
#         if type(cnt[0][0]) != int and type(cnt[0][0]) != float:
#             print("Error: Contour points must be integers or floats.")
#         else:
#             # 计算轮廓面积并添加到列表中
#             areas.append((i, cv2.contourArea(cnt)))

#     #areas.append((i,cv2.contourArea(cnt)))

# a2=sorted(areas,key=lambda d:d[1],reverse=True)

# '''
# for i,are in a2:
#     if are>100:
#         continue
#     cv2.drawContours(img22,contours,i,(0,0,255),3)
#     print(i,are)

#     cv2.imshow('drawContours',img22)
#     cv2.waitKey(0)
# #cv2.destroyAllWindows()
# '''

# #截取原图，把长方形纠正
# cnt=contours[0]
# print(cnt)
# hull=cv2.convexHull(cnt)
# epsilon=0.001*cv2.arcLength(hull,True)
# simplified_cnt=cv2.approxPolyDP(hull,epsilon,True)

# epsilon=0.1*cv2.arcLength(cnt,True)
# approx=cv2.approxPolyDP(cnt,epsilon,True)
# print(approx)
# cv2.drawContours(img22,[arpprox],0,(255,0,0),3)
# cv2.imshow('approxPolyDP',img22)
# cv2.waitKey(0)
# exit(3)

# '''
# findHomography(srcPoints,dstPints,method=None,ransacReprojThreshold=None,mask=None,maxIters=None,confidence=None)
# H=cv2.findHomography(srcPoints=cnt.astype('single'),dstPoints=np.array([[[0.,0.]],[[2150.,0.]],[[2150.,2800.]],[[0.,2800.]]]))
# M,mask=cv2.findHomography(src_pts,dst_pts,cv2.RANSAC,5.0)

# now that we have our screen contour, we need to determine 
# the top-left,top-right,bottom-right,bottom-left
# points so that we can later warp the iamge--we'll start
# by reshaping our contour to be our finals and initializing
# our ouput rectangle in top-left,top-right,bottom-right
# and bottom-left order
# '''

# pts=approx.reshape(4,2)
# rect=np.zeros((4,2),dtype='float32')

# #the top-left point has the smallest sum whereas the
# #bottom-righthas the largest sum
# s=pts.sum(axis=1)
# rect[0]=pts[np.argmin(s)]
# rect[1]=pts[np.argmax(s)]

# #compute the difference between the points -- the top-right
# #will have the minumum difference and the bottom-left will
# #have the maximum difference
# diff=np.diff(pts,axis=1)
# rect[1]=pts[np.argmin(diff)]
# rect[3]=pts[np.argmax(diff)]

# #multiply the rectangle by the original ratio
# ratio=image.shape[0]/300.0
# rect*=ratio

# #now that we have our rectangle of points,let's compute
# #the width of our new images
# (tl,tr,br,bl)=rect
# widthA=np.sqrt(((br[0]-bl[1])**2)+((br[1]-bl[1])**2))
# widthB=np.sqrt(((tr[0]-tl[0])**2)+((tr[1]-tl[1])**2))

# #……and now for the height of our new image
# heightA=np.sqrt(((tr[0]-br[0])**2)+((tr[1]-br[1])**2))
# heightB=np.sqrt(((tl[0]-bl[1])**2)+((tl[1]-bl[1])**2))

# #take the maximum of the width and height values to reach
# #our final dimensions
# maxWidth=max(int(widthA),int(widthB))
# maxHeight=max(int(heightA),int(heightB))

# #contruct our destination points which will be used to
# #map the screen to a top-down,"birds eye"view

# dst=np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]],dtype="float32")

# #caculate the perspective transform matrix and warp
# #the perspective to grab the screen
# M=cv2.getPerspectiveTransform(rect,dst)
# warp=cv2.warpPerspective(img22,M,(maxWidth,maxHeight))

# #final_image=cv2.warpPerspective(img22,H,(2150,2800))

# cv2.imshow('final_image',warp)
# cv2.waitKey(0)

#图像相减3
# def diff(img,img1): #return just the difference of the two images
#     return cv2.absdiff(img,img1)

# def diff_remove_bg(img0,img,img1): #remove the background but requires three images
#     d1=diff(img0,img)
#     d2=diff(img,img1)
#     return cv2.bitwise_and(d1,d2)

# #img1=cv2.imread('subtract1.jpg')
# img1=cv2.imread('subtract1.jpg',0) #灰度图
# #img2=cv2.imread('subtract2.jpg')
# img2=cv2.imread('subtract2.jpg',0)

# cv2.imshow('subtract1.jpg',img1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow('subtract2.jpg',img2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #
# st=diff_remove_bg(img2,img1,img2)
# cv2.imshow('after subtract',st)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #图像相减
# cap=cv2.VideoCapture(0)
# ret=cap.set(3,640)
# ret=cap.set(4,480)

# cap.read()

# '''
# cal=[cap.red()[1]for x in range(20)]

# #mean之间诶的加减是不行的
# bgimg0=np.mean(np.sum(cal))
# bgimg0=np.average(cal)
# bgimg0=np.mean(cal)
# nps1=sum(cal)
# mean1=nps1/len(cal)
# #mean1[mean1<0]=0
# #mean1[mean>255]=255
# cv2.imshow('bgimg',mean1)
# cv2.waitKey(0)
# exit(3)
# '''

# frame_no=100
# #cap.set(1,frame_no)#第10帧
# ret,bgimg0=cap.read()#背景
# bgimg=cv2.cvtColor(bgimg0,cv2.COLOR_BGR2GRAY)  
# cv2.imshow('bgimg'+str(frame_no),bgimg0)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# while cap.isOpened():
#     ret,frame=cap.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     st=cv2.subtract(gray,bgimg)

#     ret,threshold=cv2.threshold(st,50,255,cv2.THRESH_BINARY)
#     image,contours=cv2.findContours(threshold,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#     print("contours size:",len(contours))

#     img=cv2.drawContours(st,contours,-1,(255,255,255),3)

#     for cnt in contours:
#         area=cv2.contourArea(cnt)
#         if area<200:
#             continue

#         peri=cv2.arcLength(cnt,True)
#         approx=cv2.approxPolyDP(cnt,0.04*peri,True)
#         if len(approx)==4:
#             (x,y,w,h)=cv2.boundingRect(approx)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

#     cv2.imshow("frame",frame)
#     cv2.imshow("subtract",img)
#     cv2.moveWindow('subtract',y=bgimg.shape[0],x=0)
#     cv2.imshow('threshold',threshold)
#     cv2.moveWindow('threshold',x=bgimg.shape[1],y=0) 
    
#     key=cv2.waitKey(delay=1)
#     if key==ord("q"):
#         break
#     elif key==ord("s"):
#         cv2.imwrite('poker=threshold.jpg',threshold)

# cv2.destroyAllWindows()

# #长方形1
# img22=cv2.imread('subtract2.jpg')

# #src_pts=np.array([[8,136],[415,52],[420,152],[12,244]],dtype=np.float32)

# src_pts=np.array([[[97,390],[210,373],[183,199],[69,214]]],dtype=np.float32)

# dst_pts=np.array([[0,0],[50,0],[50,100],[0,100]],dtype=np.float32)

# M=cv2.getPerspectiveTransform(src_pts,dst_pts)
# warp=cv2.warpPerspective(img22,M,(50,100))

# cv2.imshow("src",img22)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imshow("warp",warp)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# #斑点检测
# def get_euler_distance(pt1,pt2):
#     return ((pt1[0]-pt2[0])**2+(pt1[1]-[pt2[1]])**2)**0.5

# img22=cv2.imread('subtract2.jpg')

# #src_pts=np.array([[8,136],[415,52],[420,152],[14,244]],dtype=np.float32)

# src_pts=np.array([[[98,390],[210,373],[183,199],[69,214]]],dtype=np.float32)
# #src_pts=np.array([[[210,373],[183,199],[69,214],[97,390]]],dtype=np.float32)

# width=get_euler_distance(src_pts[0][0],src_pts[0][1])
# height=get_euler_distance(src_pts[0][0],src_pts[0][3])

# dst_pts=np.array([[0,0],[width,0],[width,height],[0,height]],dtype=np.float32)

# M=cv2.getPerspectiveTransform(src_pts,dst_pts)
# warp=cv2.warpPerspective(img22,M,(int(width),int(height)))

# warp=cv2.flip(warp,flipCode=1)

# cv2.imshow('src',img22)
# cv2.imshow('warp',warp)
# cv2.waitKey(0)
