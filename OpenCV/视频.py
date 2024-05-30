# from skimage.measure import compare_ssim as compare_ssim
# from kimage.measure import compare_mse as mse
# import matplotlib.pyplot as plt
import numpy as np
import cv2
import string,random
#from utils import mse

# cap=cv2.VideoCapture(0)

# #frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)#4.720
# #frame_width=cap.get(cv2.CAP_PROP_ FRAME_WIDTH)#3 ,1280
# #frame_height=int(480/frame_width*frame_height)# 270

# #ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)#高
# #ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)

# ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
# ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)

# title='camera compare'
# plt.ion()

# #cap.read()
# ret,frame=cap.read()
# temp=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# #TODO 前10帧
# while cap.isOpened():
#     ret,frame=cap.read()
#     gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     m=mse(temp,gray)
#     s=ssim(temp,gray)
#     print('MSE:%.2f,SSIM:%.2f'%(m,s))

#     temp=gray.copy()
#     continue

#     ## setup the figure
#     #fig=plt.figure(title)
#     #plt.subtitle9("MSE:%.2f,SSIM:%.2f"%(m,s))
    
#     #show first image
#     #ax=fig.add_subplot(1,2,1)
#     #plt.imshow(temp,cmap=plt.cm.gray)
#     #plt.axis('off')

#     #show the second image
#     #ax=fig.add_subplot(1,2,2)
#     #plt.imshow(gray,cmap=plt.cm.gray)
#     #plt.axis('off')

#     #show the image
#     #plt.show()

# def mse(iamgeA,imageB):
#     #the 'Mean Squared Error'between the two images is the sum of the squared difference between the two images;
#     #NOTE: the two images must have the same dimension
#     err=np.sum((iamgeA.astype('float')-imageB.astype('float'))**2)
#     err/=float(iamgeA.shape[0]*iamgeA.shape[1])

#     #return the MSE,the lower the error ,the more 'similar'
#     #the two images are
#     return err

# cap=cv2.VideoCapture(0) #一般的笔记本有内置摄像头。所以参数就是0.可以设置成1或者其他参数来选择摄像头来

# '''
# 你可以使用函数cap.get(proId)来获得一些参数信息
# propId可以是0到18之间的任何整数

# 其中的一些值可以使用cap.set(propId,value)来修改value的值
# 例如，我可以使用cap.get(3) cv2.CAP_PROP_FRAME_WIDT和cap.get(4)cv2.CAP_PROP_FRAME_HEIGHT来查看每一帧的宽和高
# 默认情况下得到的值是640*480，但是可以使用ret=cap.set(3,320)和ret=cap.set(4,240)来把宽和高改称320*240
# '''

# #ret=cap.set(3,320)
# #ret=cap.set(4,240)

# #ret=cap.set(cv2.CAP_RPOP_FRAME_WIDTH,480)#避免计算量过大
# #ret=cap.set(cv2.CPA_PROP_FRAME_HEIGHT,270)

# #等比缩放
# frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# frame_height=int(480/frame_width*frame_height)
# ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frame_height)#高
# ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,480)

# #while(True):
# while cap.isOpened(): #检查是否成功初始化，否则就使用函数cap.opened()
#     #Capture frame-by-frame
#     ret,frame=cap.read()#ret返回一个布尔值
#     #print('frame shape:',frame.shape)#(720,1280,3)

#     frame=cv2.flip(frame,flipCode=1)#左右翻转，使用笔记本摄像头才有用
#     #flipCode:翻转方向：1、水平翻转；0：垂直翻转；-1：水平垂直翻转

#     #Our operation on the frame come here
#     #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     #Display the resulting frame
#     cv2.imshow('frame',frame)
#     cv2.setWindowTitle('frame','COLOR_BGR2GRAY')

#     #Property=cv2.getWindowProperty('frame',0)
#     #if cv2.waitKey(1)&0xFF==ord('q')#不行
#     #   break
#     key=cv2.waitKey(delay=10)
#     if key==ord('q'):
#         break

# #When everything done,release the capture
# cap.release()
# cv2.destroyAllWindows()

# cap=cv2.VideoCapture('video.mp4')
# #cap=cv2.VideoCapture('output.avi')
# #cap=cv2.VideoCapture('Mnions_banana.mp4)

# #帧率
# fps=cap.get(cv2.CAP_PROP_FPS)
# print("Frame per second using video.get(cv2.CAP_PROP_FPS):{0}".format(fps))
# #总共有多少帧
# num_frames=cap.get(cv2.CAP_PROP_FRAME_COUNT)
# print('共有',num_frames,'帧')

# frame_height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
# frame_width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
# print('高:',frame_height,'宽:',frame_width)

# FRAME_NOW=cap.get(cv2.CAP_PROP_POS_FRAMES)#第0帧
# print('当前帧数',FRAME_NOW)#当前帧数0.0


# #读取指定帧
# frame_no=121
# cap.set(1,frame_no)#Where frame_no is the frame you want
# ret,frame=cap.read() #Read the frame
# cv2.imshow('frame_no'+str(frame_no),frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# FRAME_NOW=cap.get(cv2.CAP_PROP_POS_FRAMES)
# print('当前帧数',FRAME_NOW)


# while cap.isOpened():
#     ret,frame=cap.read()
#     FRAME_NOW=cap.get(cv2.CAP_PROP_POS_FRAMES)#当前帧数
#     print('当前帧数',FRAME_NOW)

#     #gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     cv2.imshow('frame',frame)
#     key=cv2.waitKey(1)
#     if key==ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

# cap=cv2.VideoCapture(0)
# width=640
# ret=cap.set(3,width)
# height=400
# ret=cap.set(4,height)

# #Define the codec and create VideoWriter object
# fourcc=cv2.VideoWriter_fourcc(*'XVID')#opencv 3.0
# #Error:'module'object has no attribute 'VideoWriter_fourcc'
# #fourcc=cv2.VideoWriter_forucc('X','V','I','D')

# out=cv2.VideoWriter('output.avi',fourcc,20.0,(width,height))

# while cap.isOpened():
#     ret,frame=cap.read()
#     if ret is True:
#         frame=cv2.resize(frame,(640,480))

#         #write the flipped frame
#         out.write(frame)

#         cv2.imshow('frame',frame)
#     else:
#         break
    
#     key=cv2.waitKey(1)
#     if key==ord('q'):
#         break

# #Release everything if job is finished
# cap.release()
# out.release()
# cv2.destroyAllWindows()

# def id_generator(size=6,chars=string.ascii_uppercase+string.digits):
#     return ''.join(random.choice(chars)for _ in range(size))

# cap=cv2.VideoCapture(0)

# while cap.isOpened():
#     ret,frame=cap.read()
#     print('frame.shape:',frame.shape)#(720,1280,3)

#     cv2.imshow('frame',frame)

#     key=cv2.waitKey(delay=1)
#     if key==ord('q'):
#         break
#     elif key==ord('s'):
#         cv2.imwrite(id_generator()+'.jpg',frame)

# cap.release()
# cv2.destroyAllWindows()

cap0=cv2.VideoCapture(0)
cap1=cv2.VideoCapture(1)
ret=cap0.set(3,320)
ret=cap0.set(4,240)
ret=cap1.set(3,320)
ret=cap1.set(4,240)

while cap0.isOpened() and cap1.isOpened():
    ret0,frame0=cap0.read()
    ret1,frame1=cap1.read()

    if ret0:
        cv2.imshow('frame0',frame0)
        cv2.setWindowTitle('frame0','On Top')
    if ret1:
        cv2.imshow('frame1',frame1)
        cv2.moveWindow('frame1',x=320,y=40)

    key=cv2.waitKey(delay=2)
    if key==ord('q'):
        break

#When everything done,release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()