import cv2
import numpy as np

# #mouse callback function

# def draw_circle(event,x,y,flags,param): #只用过一件事：在双击过的地方绘制一个圆圈
#     if event==cv2.EVENT_LBUTTONDBLCLK:
#         cv2.circle(img,(x,y),100,(255,0,0),-1)
    
# #创建图像与窗口与回调函数绑定
# img=np.zeros((512,512,3),np.uint8)
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)

# while True:
#     cv2.imshow('image',draw_circle)
#     #if cv2.waitKey(20) & 0xFF==27:
#     #   break
#     key=cv2.waitKey(1)
#     if key==ord("q"):
#         break

# cv2.destroyAllWindows()

# #当鼠标按下时变为True
# drawing=False
# #如果 mode 为 true 绘制矩形。按下'm'变成绘制曲线。mode=True
# ix,iy=-1,-1

# #创建回调函数(回调函数包含两部分，一部分画矩形，一部分画圆圈)
# def draw_circle(event,x,y,flags,param):
#     global ix,iy,drawing,mode
#     #当按下左键是返回起始位置坐标
#     if event==cv2.EVENT_LBUTTONDOWN:
#         drawing=True
#         ix,iy=x,y
#     #当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
#     elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
#         if drawing is True:
#             if mode is True:
#                 cv2.rectangle(img,(ix,iy,(x,y),(0,255,0),-1))
#             else:
#                 #绘制圆圈，小原点连在一起成立线，3代表了笔画粗细
#                 cv2.circle(img,(x,y),3,(0,0,255),-1)
#                 #下面注释调的代码是起始点为原点，起点到终点为半径的
#                 #r=int(np.sqrt((x-ix)**2)+(y-iy)**2)
#                 #cv2.circle(img,(x,y),r,(0,0,255),-1)
#     elif event==cv2.EVENT_LBUTTONUP: #当鼠标松开停止绘画
#         drawing=True
#         #if mode==True:
#         #   cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
#         #else:
#         #   cv2.circle(img,(x,y),5,(0,0,255),-1)

# img=np.zeros((512,512,3),np.uint8)
# mode=False

# while True:
#     cv2.imshow('image',img)
#     k=cv2.waitKey(1)
#     if k==ord('m'):
#         mode=not mode
#     elif k==ord('q'):
#         break

# cv2.destroyAllWindows()

def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,y)
    
    if event==cv2.EVENT_RBUTTONDBLCLK:
        red=img[y,x,2]
        blue=img[y,x,0]
        green=img[y,x,1]
        print(red,green,blue)

        strRGB=str(red)+","+str(green)+","+str(blue)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,strRGB,(x,y),font,1,(255,255,255),2)
        cv2.imshow('original',img)

img=cv2.imread('1.jpeg')
cv2.imshow('orginal',img)

cv2.setMouseCallback('original',click_event)
cv2.waitKey(0)
cv2.imwrite('putText.jpg',img)

cv2.destroyAllWindows()