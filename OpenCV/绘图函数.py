from time import sleep
import cv2
import numpy as np
import math

# def click_event(event,x,y,flags,param):
#     '''
#     用左键点击屏幕，打印坐标
#     param event:
#     param x
#     param y
#     param flags
#     param param
#     return 
#     '''
#     if event==cv2.EVENT_LBUTTONDOWN:
#         print(x,y,flags,pram)

# cv2.namedWindow('Canvas',cv2.WINDOW_GUI_EXPANDED)
# cv2.setMouseCallback('Cavas',click_event)

# canvas=np.zeros((300,300,3),dtype="uint8")
# while True:
#     try:
#         for i in range(0,25):
#             radius=np.random.randint(5,high=200)
#             color=np.random.randint(0,high=256,size=(3,)).tolist()
#             pt=np.random.randint(0,high=300,size=(2,))  
#             cv2.circle(canvas,tuple(pt),radius,color,-1) 
        
#         cv2.imshow("Canvas",canvas)    

#         key=cv2.waitKey(0) #等待1秒
#         if key==ord('q'):
#             break
#         else:
#             #sleep(1)
#             continue
#     except KeyboardInterrupt as e:
#         print('KeyboardInterrupt',e)
#     finally:
#         cv2.imwrite('random-circles2.jpg',canvas)
    
# r1=70
# r2=30

# ang=60

# d=170
# h=int(d/2*math.sqrt(3))

# dot_red=(256,128)
# dot_green=(int(dot_red[0]-d/2),dot_red[1]+h)
# dot_blue=(int(dot_red[0]+d/2),dot_red[1]+h)

# #tan=float(dot_red[0]-dot_green[0])/(dot_green[1]-dot_red[0])
# #ang=math.aten(tan)/math.pi*180

# red=(0,0,255)
# green=(0,255,0)
# blue=(255,0,0)
# black=(0,0,0)

# full=-1

# img=np.zeros((512,512,3),np.uint8)
# #img=np.ones((512,512,3),np.uint8)

# cv2.circle(img,dot_red,r1,red,full)
# cv2.circle(img,dot_green,r2,green,full)
# cv2.circle(img,dot_blue,r1,blue,full)
# cv2.circle(img,dot_red,r2,black,full)
# cv2.circle(img,dot_green,r2,black,full)
# cv2.circle(img,dot_blue,r2,black,full)

# cv2.ellipse(img,dot_red,(r1,r1),ang,0,black,full)
# cv2.ellipse(img,dot_green,(r1,r1),360-ang,0,ang,black,full)
# cv2.ellipse(img,dot_blue,(r1,r1),360-2*ang,ang,0,black,full)

# font=cv2.FONT_HERSHEY_SIMPLEX
# cv2.putText(img,text='OpenCV',org=(15,450),fontFace=font,fontScale=4,color=(255,255,255),thickness=10)

# cv2.imwrite("opencv_logo.png",img)
# cv2.imwrite("opencv_logo2.png",img)

# img=np.zeros((100,300,3),dtype=np.uint8)

# ft=cv2.freetype.createFreeType2()
# #ft.loadFontData(fontFileName="",id=0)
# #ft.loadFontData(fontFileName="",id=0)

# ft.loadFontData(fontFileName='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', id=0)

# ft.putText(img=img,text='你好中文',org=(15,70),fontHeight=60,color=(255,255,255),thickness=-1,line_type=cv2.LINE_AA,bottomLeftOrgin=True)

# cv2.imshow('freetype',img)
# cv2.waitKey(0)

'''
img:图像
color:颜色
thickness:线条粗细
linetype:线条的类型。cv2.LINE_AA为抗锯齿，这样看起来会非常平滑
'''

#Create a black image
img=np.zeros((512,512,3),np.uint8)

#Draw a diagonal blue line with thickness of 5px
cv2.line(img,pt1=(0,0),pt2=(511,511),color=(255,0,0),thickness=5)
#cv2.polylines()可以用来画很多条线。只需要把想画的线放在一个列表中，将列表传给函数就可以了。每条线会被独立绘制。这会比用cv2.line()一条一条的绘制要快一些
cv2.arrowedLine(img,pt1=(21,13),pt2=(151,401),color=(255,0,0),thickness=5)

cv2.rectangle(img,(384,0),(510,128),(0,255,0),3)

cv2.circle(img,center=(447,63),radius=63,color=(0,0,255),thickness=-1)

#一个参数是中心点的位置坐标，下一个参数是长轴和短轴的长度。椭圆沿逆时针方向旋转的角度
#椭圆弧沿顺时针方向其实的角度和结束角度，如果是0和360就是整个椭圆
cv2.ellipse(img,center=(256,256),axes=(100,50),angle=0,startAngle=0,endAngle=180,color=255,thickness=-1)

pts=np.array([[10,5],[20,30],[70,20],[50,10]],np.int32)
pts=pts.reshape((-1,1,2))

font=cv2.FONT_HERSHEY_SIMPLEX
#org:Bottom-left corner of the text string in the image
#或使用bottomLeftOrgin=True,文字会上下颠倒
cv2.putText(img,text="bottomLeftOrigin",org=(10,400),fontFace=font,fontScale=1,color=(255,255,255),thickness=1,bottomLeftOrigin=True)
cv2.putText(img,text='OpenCV',org=(10,500),fontFace=font,fontScale=4,color=(255,255,255),thickness=2)

#所有的绘图函数的返回值都是None

winname='example'
cv2.namedWindow(winname,0)
cv2.imshow(winname,img)

cv2.imwrite('example.png',img)


