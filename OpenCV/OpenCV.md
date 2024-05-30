Define the codec and create VideoWriter object
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

    if ret is True:
        frame=cv2.flip(frame,0)

        #write the flipped frame
        out.write(frame)

#安装合适版本的ffmpeg或者gstreamer。

# 保存视频
创建一个VideoWrtier的对象。我们应该确定一个输出文件的名字。接下来指定FourCC编码。播放频率和帧的大小也都需要确定。最后一个是isColor标签。如果是True，每一帧就是彩色图，否则就是灰度图。
FourCC就是一个4字节码，用来确定是普宁的编码格式。可用编码列表可以从fourcc.org茶道

# 把鼠标当画笔
cv2.setMouseCallback()

import cv2
import numpy as np
#mouse callback function

def draw_circle(event,x,y,flags,param):
    if evetn=cv2.EVENT_LBUTTONDBLICK:
    cv2.circle(img,(x,y),100,(255,0,0),-1)

    #创建图像并将窗口与回调函数绑定 
    img=np.zeros((512,512,3),np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',img)
        if cv2.waitKey(0):
            break
    cv2.destroyAllWindows()


import cv2
import numpy as np

#当鼠标按下时变为True
drawing=False
#如果mode为True绘制矩形。按下'm'变成绘制曲线
mode=True
ix,iy=-1,-1

# 创建回调函数
def draw_circle(event,x,y,flags,param):
    global ix,iy,drawing,mode
    #当按下左键是返回起始位置坐标
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    #当鼠标左键按下并移动是绘制图形。event可以查看移动，flag查看是否按下
    elif event==cv2.EVENT_MOUSEMOVE and flags=cv2.EVENT_FLAG_BUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),(0,255,0),-1)
            else:
            #绘制圆圈，小圆点在以其就成了线，3代表画笔的粗细
                cv2.circle(img,(x,y),r,(0,0,255),-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing==False


#我们要把回调函数与OpenCV窗口绑定在以其。在主循环中我们将m与模式转换绑定在一起
img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('image',img)
    k=cv2.waitKey(1)
    if k==ord('m'):
        mode=not mode
    elif k==27:
        break

# 用滑动条做调色板
cv2.getTrackbarPos(),cv2.creatTrackbar()
cv2.getTrackbarPos()函数第一个参数是滑动条的名字，第二个参数是滑动条被放置窗口的名字，第三个参数是滑动条的默认位置。第四个参数是滑动条的最大值，第五个参数是回调函数，每次滑动条的滑动都会调用回调函数。回调函数通常都会含有一个默认参数，就是滑动条的位置
滑动条的另一个重要应用就是用作转换按钮。默认情况下，OpenCV本身不带有转换按钮，只有当转换按钮只向ON时，滑动条的滑动才有用，否则窗口都是黑的

import cv2
import numpy as np

def nothing(x):
    pass

#创建一副黑色图像
img=np.zeros((300,512,3),np.uint8)
cv2.namedWindow('images')

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)

switch='0:OFF\n1:ON'
cv2.createTrackbar(switch,'image',0,1,nothing)

while(1):
   cv2.imshow('image',img)
   k=cv2.waitKey(1)
   if k==27:
        break
    
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G','image')
    b=cv2.getTrackbarPos('B','image')
    s=cv2.getTrackbarPos(switch,'image')

    if s=0:
        img[:]=0
    else:
        img[:]=[b,g,r]
    
cv2.destroyAllWindows()

import cv2
import numpy as np

def nothing(x):
    pass

drawing=False
mode=True
ix,iy=-1,-1

def draw_circle(event,x,y,flags,param):
    r=cv2.getTrackbarPos('R','image')
    g=cv2.getTrackbarPos('G',image)
    b=cv2.getTrackbarPos('B','image')
    color=(b,g,r)

    global ix,iy,drawing,mode
    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y
    elif event==cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:
        if drawing==True:
            if mode==True:
                cv2.rectangle(img,(ix,iy),(x,y),color,-1)
            else:
                cv2.circle(img,(x,y),3,color,-1)
    elif event==cv2.EVENT_LBUTTONUP:
        drawing==False

img=np.zeros((512,512,3),np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)
cv2.setMouseCallback('image',draw_circle)
while(1):
    cv2.imshow('iamge',img)

    k=cv2.waitKey(0)
    if k==ord('m'):
        mode=not mode
    elif k==27:
        break

# 图像的基础操作
## 目标
获取像素值并修改
获取图像的属性(信息)
图像的ROI()
图像通道的拆分及合并

#读入图像
import cv2
import numpy as np
img.imread('')

#根据像素的行和列的坐标获取他的像素值
import cv2
import  numpy as np
img=cv2.imread('')

px=img[100,100]
print px
blue=img[100,100,0]
print blue

#修改像素值
import cv2
import numpy as np
img=cv2.imread('')
img[100,100]=[255,255,255]
print img[100,100]

#上面体的哦啊的方法被用来选取矩阵的一个区域，比如说前5行的后3列。对于获取每一个像素值，也许使用Numpy的array.item()和array.itemset()会更好。但是返回值是标量。如果你想获得所有B，G，R的值，你需要使用array.item()分割他们
import cv2
import numpy as np
img=cv2.imread('')
print img.item(10,10,2)
img.item.set((10,10,2),100)
print img.item(10,10,2)

#获取图像属性
图像属性包括:行、列，通道，图像数据类型，像素数目等
img.shape可以获取图像的形状。他的返回值是一个包含行数，列数，通道数的元组
import cv2
import numpy as np
img=cv2.imread('')
print img.shape

img.size可以返回图像的像素数目
import cv2
import numpy as np
img=cv2.imread('')
print img.size

img.dtype返回的是图像的数据类型
import cv2
import numpy as np
img=cv2.imread('')
print img.dtype

#在除虫(debug)时，img.dtype非常重要。因为在OpenCV-Python代码中经常出现数据类型的不一致

# 图像ROI
有时你需要对一幅图像的特定区域进行操作。
ROI也是使用Numpy索引来获得的。现在我们选择球的部分并把他拷贝到图像的其他区域
import cv2
import numpy as np
img=cv2.imread('')
ball=img[280:340,330:390]
img[273:333,100:160]=ball

# 拆分及合并图像通道
有时我们需要对BGR三个通道分别进行操作。这时就需要把BGR拆分成单个通道。有时需要把独立通道的图片合并成一个BGR通道
import cv2
import numpy as np
img=cv2.imread('')
b,g,r=cv2.split(img)
img=cv2.merge(b,g,r)

import cv2
import numpy as np
img=cv2.imread('')
b=img[:,:,2]=0
#cv2.split()是一个比较耗时的操作。只有真正需要时才用他，能用Numpy索引就尽量用

# 为图像扩边(填充)
如果你想在图像周围创建一个像相框一样的边，可以使用cv2.copyMakeBorder()函数。这经常在卷积运算或0填充时被用到
src输入图像
top,bottom,left,right对应边界的像素数目
borderType要添加的类型边界
    cv2.BORDER_COSTANT添加有颜色的常数值边界，还需要下一个参数(value)
    cv2.BORDER_REFLECT边界元素的镜像。比如fedcba|abcde-fgh|hgfedcb
    cv2.BORDER_REFELCT_101 or cv2.BORDER_DEFAULT跟上面一样，但稍作改动gfedcb|abcdefgh|gfedcba
    cv2.BORDER_REPLICATE重复最后一个元素。例如:aaaaaa|abcdefgh|hhhhhhh
    cv2.BORDER_WRAP,例如cdefgh|abcdefgh|abcdefg
    value边界颜色，如果边界的类型是cv2.BORDER_CONSTRANT

import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE=[255,0,0]
img1=cv2.imread('')
replicate=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REPLICATE)
reflect=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT)
reflect101=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_REFLECT_101)
wrap=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_WRAP)
constant=cv2.copyMakeBorder(img1,10,10,10,10,cv2.BORDER_CONSTANT,value=BLUE)

plt.subplot(231),plt.imshow(img1,'gray'),plt.title('ORIGINAL')
plt.subplot(232),plt.imshow(replicate,'gray'),plt.title('REPLICATE')
plt.subplot(233),plt.imshow(reflect,'gray'),plt.title('REFLECT')
plt.subplot(234),plt.imshow(reflect101,'gray'),plt.title('REFLECT_101')
plt.subplot(235),plt.imshow(wrap,'gray'),plt.title('WRAP')
plt.subplot(236),plt.imshow(constant,'gray'),plt.title('CONSTANT')
plt.show()

# 图像上的算术运算
学习图像上的算术运算，加法，减法，位运算等
我们将要要学习的函数有：cv2.add()，cv2.addWeighted()等

#图像加法(两幅图像进行加法运算，大小，类型必须一致)
#OpenCV中的加法与Numpy的加法是有所不同的。OpenCV的加法是一种饱和操作，而Numpy的加法是一种模操作

x=np.uint8([250])
y=np.uint8([10])
print(cv2.add(x,y))
print x+y

#图像混合
cv2.addWeighted()
import cv2
import numpy as np

img1=cv2.imread('')
img2=cv2.imread('')

dst=cv2.addWeighted(img1,0.7,img2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

#按位运算
and,or,not,xor等
当我们提取图像的一部分，选择非矩形ROI时这些操作会很有用。
import cv2
import numpy as np

#加载图像
img1=cv2.imread()
img2=cv2.imread()

rows,cols,channels=img2.shape
roi=img1[0:rows,0:cols]

img2gray=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
ret,mask=cv2.threshold(img2gray,175,255,cv2.THRESHOLD_BINARY)
mask_inv=cv2.bitwise_not(mask)

img1_bg=cv2.bitwise_and(roi,roi,mask=mask)
img2_bg=cv2.bitwise_and(img2,img2,mask=mask_inv)

dst=cv2.add(img1_bg,img2_bg)
img1[0:rows,0:cols]=dst

cv2.imshow('res',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 程序性能检测及优化
检测程序的效率
一些能够提高程序效率的技巧
cv2.getTickCount()
cv2.getTickFrequency()
Python也提供了一个较time的模块，另一个叫做profile的模块会帮助你得到一份关于你程序的详细报告，其中包含了每个函数运行需要的时间，以及每个函数被调用的此书。如果你正在使用IPython的话，所有这些特点都以一种用户友好的方式整合在一起了。

cv2.getTickCount函数返回从参考点到这个函数被执行的时钟数。所以当你在一个函数执行前后都调用的话，你就会得到这个函数的执行时间(时钟数)
cv2.getTickFrequency返回时钟频率，或者每秒钟的时钟数。

import cv2
import numpy as np

e1=cv2.getTickCount()
#your code execution
e2=cv2.getTickCount()
time=(e2-e1)/cv2.getTickFrequency()

用窗口大小不同(5,7,9)的核函数来做中值滤波
import cv2
import numpy as np

img1=cv2.imread()

e1=cv2.getTickCount()
for i in xrange(5,49,2):
    img1=cv2.medianBlur(img1,i)
e2=cv2.getTickCount()
t=(e2-e1)/cv2.getTickFrequency()
print t

#你也可以用time模块实现上面的功能。但是要用time.time()而不是cv2.getTickCount。

# OpenCV中的默认优化
可以使用函数cv2.useOptimized()来查看优化是否被开启
可以使用函数cv2.setUseOptimized()来开启优化

import cv2
import numpy as np

cv2.useOptimized()

res=cv2.meidianBlur(img,49)

cv2.setUseOptimized(False)

# 在IPython中检测程序效率
要比较两个相似操作的效率，这是你可以使用IPython为你提供的魔法命令%time。她会让代码运行好几次从而得到一个准确的运行时间。也是可以用来测试单行代码的

Python的标量计算比Numpy的标量计算要快。对于仅包含一两个元素的操作Python标量比Numpy的数组快，但数组稍大一点Numpy胜出

cv2.countNoneZero()和np.count_nonzero()

import cv2
import numpy as np

%timeit z=cv2.countNoneZero(img)

%timeit z=np.count_nonzero(img)

#当使用Numpy对试图(而非复制)进行操作时

# 更多IPython的魔法命令
profiling,line profiling,内存使用等


# 转换颜色空间
cv2.cvtColor(),cv2.inRange()
cv2.cvtColor(input_image,flag)，其中flag时转换类型

得到所有可用的flag

import cv2
flags=[i for in dir(cv2) if i startswith('COLOR_')]
print flags


在OpenCV的HSV格式中，H(色彩/色度)的取值范围是[0,179],S(饱和度)的取值范围[0,255],V(亮度)的取值范围[0,255]，不同的软件使用值可能不同。当你需要拿OpenCV的HSV值与其他软件的HSV值进行对比时，一定要记得归一化

# 物体跟踪
在HSV颜色空间中要比在BGR空间中更容易表示一个特定颜色。要提取的时一个蓝色的物体步骤
从视频中获取每一帧的图像
将图像转换到HSV空间
设置HSV阈值到蓝色范围
提取蓝色物体

import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()

    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #设置蓝色阈值
    lower_blue=np.array([110,50,50])
    upper_blue=np.array([120,255,255])

    #根据阈值构建掩膜
    mask=cv2.inRange(hsv,lower_blue,upper_blue)

    #对原图像进行位运算
    res=cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(0)
    if k==27:
        break
cv2.destroyAllWindows()
图像中仍然有一些噪音
这是物体跟踪中最简单的方法，当你学习了轮廓之后，就可以找到物体的中心，并根据重心来跟踪物体，仅仅在摄像头前挥挥手就可以画出同的图形

# 怎样找到要跟踪对象的HSV值
cv2.cvtColor可以用到这里。
找到绿色的HSV值
import cv2
import numpy as np

<!-- green=np.uint8([0,255,0])
hsv_green=cv2.cvtColor(green,cv2.COLOR2HSV) -->

green=np.uint8([[[0,255,0]]])
hsv_green=cv2.cvtColor(green,cv2.COLOR_BGR2HSV)
print hsv_green

现在你可以分别用[H-100,100,100]和[H+100,255,255]做上下阈值。除了这个方法之外，你可以使用任何其他图像编辑软件(例如GIMP)或者在线转换软件来找到相应的HSV值，但是最后别忘了调节HSV的范围

# 几何变换
学习对图像进行各种几何变换，例如移动，旋转，仿射变换等
cv2.getPerspectiveTransform
OpenCV提供了两个变换函数,cv2.warpAffine和cv2.warpPerspective，使用这两个函数你可以时线所有类型的变换
cv2.warpAffine接收的参数时2*3的变换矩阵
cv2.warpPerspective接受的参数是3*3的变换矩阵

# 扩展缩放
图像的尺寸可以自己手动设置，你也可以指定缩放因子。可以选择不同的插值方法。
在缩放时推荐时用cv2.INTER_AREA
扩展时推荐时用cv2.INTER_CUBIC和cv2.INTER_LINEAR
默认情况下所有改变图像尺寸大小的操作时用的插值方法都是cv2.INTER_LINEAR

import cv2
import numpy as np

img=cv2.imread()
res=cv2.resize(img,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

height,width=img.shape[:2]
res=cv2.resize(img,(2*width,2*height),interpolation=cv2.INTER_CUBIC)

while(1):
    cv2.imshow('res',res)
    cv2.imshow('img',img)

    if cv2.waitKey(0):
        break
cv2.destoyAllWindows()

cv2.resize(src,dst,interpolation=cv2.INTER_LINEAR)

# 平移
import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()

    #转换的到HSV
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    lower_blue=np.array([110,50,50])
    upper_blue=np.array([130,255,255])

    mask=cv2.inRange(hsv,lower_blue,upper_blue)

    res=cv2.bitwise_and(frame,frame,mask=mask)

    cv2.imshow('frame',frame)
    cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    k=cv2.waitKey(5)&0xFF
    if k==27:
        break

cv2.destroyAllWindows()

#cv2.warpAffine()的第三个参数是输出图像的大小，他的格式应该是图像的(宽、高)，应该记住的是图像的宽对应的是列数，高对应的是行数

# 旋转
import cv2
import numpy as np

img=cv2.imread('')

rows,cols=img.shape

#第一个参数为旋转中心，第二个参数为旋转角度，第三个为旋转后的缩放因子
#可以通过设置旋转中心，缩放因子，以及窗口大小来放置旋转后超出边界的问题

M=cv2.getRotationMatrix2D((cols/2,rows/2),45,0.6)

#第三个参数是输出图像的尺寸中心
dst=cv2.warpAffine(img,M,(2*cols,2*rows))
while(1):
    cv2.imshow('img',dst)
    if cv2.waitKey(1)&0xFF==27:
        break

cv2.destroyAllWindows()


# 仿射变换
在仿射变换中，原图中所有的平行线在结果图像中同样平行。为了创建这个矩阵我们需要从原图像中找到三个点以及他们在输出图像中的位置。然后cv2.getAffineTransform会创建一个2*3的矩阵，最后这个矩阵会被传给函数cv2.wrapAffine()

import cv2
import numpy as np

img=cv2.imread()
rows,cols,ch=img.shape

pts1=np.float32([[50,50],[200,50],[50,200]])
pts2=np.float32([[10,100],[200,50],[100,250]])

M=cv2.getAffineTransform(pts1,pts2)

dst=cv2.warpAffine(img,M,(cols,rows))

plt.subplot(121,plt.imshow(img),plt.title('Input'))
plt.subplot(121,plt.imshow(img),plt.title('Output'))
plt.show()

#透视变换
对于透视变换，我们需要一个3*3变换矩阵。在变换前后直线还是直线。要构建这个变换矩阵，你需要在输入图像上找4个点，以及他们在输出图像上对应的位置。这4个点钟的任意3个都不能共线。这吧额变换矩阵可以有函数cv2.getPerspectiveTransform()构建。然后把这个矩阵传给函数cv2.warpPerspective()

import cv2
import numpy
from matplotlib import pyplot as plt

img=cv2.imread()
rows,cols,ch=img.shape

pts1=np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2=np.float32([[0,0],[300,0],[0,300],[300,300]])

M=cv2.getPerspectiveTransform(pts1,pts2)

dst=cv2.warpPerspective(img,M,(300,300))

plt.subplot(121,plt.imshow(img),plt.title('Input'))
plt.subplot(121,plt.imshow(img),plt.title('Output'))
plt.show()

# 图像阈值
简单阈值，自适应阈值,Otsu's二值化阈值
cv2.threshold,cv2.adaptiveThreshold等

# 简单阈值
当像素值高于阈值时，我们给这个像素赋予一个新值(可能时白色)，否则赋予另一种颜色(也许是黑色)。这个函数就是cv2.threshold()。第一个参数为原图像，并且为灰度图。第二个参数就是用来对像素进行分类的阈值。第三个参数就是当像素值高于阈值时应该被赋予的新的像素值。OpenCV提供了多种不同的阈值方法，这是由第四个参数来决定的
cv2.THRESHOLD_BIANRY
cv2.THRESH_BINARY_INV
cv2.THRESH_TRUNC
cv2.THRESH_TOZERO
cv2.THRESH_TOZERO_INV
该函数有两个返回值，第一个为retVal，第二个为阈值化之后的结果图像了
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread()
ret,thresh1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)
ret,thresh2=cv2.threshold(img,127,255,cv2.THRESH_BIANRY_INV)
ret,thresh3=cv2.threshold(img,127,255,cv2.THRESH_TRUNC)
ret,thresh4=cv2.threshold(img,127,255,cv2.THRESH_TOZERO)
ret,thresh5=cv2.threshold(img,127,255,cv2.TREHSH_TOZERO_INV)

titles=['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images=[img,thresh1,thresh2,thresh3,thresh4,thersh5]

for i in xrange(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# 自适应阈值
在前面的部分我们时用时全局阈值，整幅图像采用同一个数作为阈值。这种方法并不适应于所有情况，尤其是当同一幅图像上的不同部分具有不同亮度时时。这种情况下我们需要采用自适应阈值。此时的阈值时根据图像上每一小区域计算与其对应的阈值。因此在同一幅图像上的不同却与采用的是不同的阈值，从而使我们能在亮度不同的情况下得到更好的结果
Adaptive Method:指定计算阈值的方法
cv2.ADAPTIVE_THRESH_MEAN_C:阈值取自相邻区域的平均值
cv2.ADAPTIVE_THRESH_GAUSSIAN_C:阈值取值相邻区域的加权和，权重作为一个高斯窗口
Block Size邻域大小(用来计算阈值的区域大小)
C一个常数，阈值就等与平均值或者加权平均值减去这个常数

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('',0)
#中值滤波
img=cv2.medianBlur(img,5)

ret,th1=cv2.threshold(img,128,255,cv2.THRESH_BINARY)
#11 为Block size,2为C值
th2=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
th3=cv2.adaptiveTrheshold(img,255,cb2.ADAPTIVE_THRESH_GAUSSIAN_C,cvf2.THRESH_BINARY,11,2)

titles=['Original Image','Global Thresholding(v=127)','Adaptive Mean Thresholding','Adaptive Gaussian Thresholding']

images=[img,th1,th2,th3]

for i in range(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

# Otsu's二值化
Otsu而值化时会用到retVal。
在使用全局阈值时，我们就随便给了一个数来做阈值，那我们怎么直到选取的数字的好坏呢，答案就是不停的尝试。
如果一幅图像是双峰图像。那就是要两个峰之间的峰谷选一个值作为ie阈值。这就是Otsu二值化需要做的。简单来说就是对一一副双峰的图像自动根据其值房图计算出一个阈值(对于非双峰图像，这种方法得到的结果可能会不理想)
这里用到的函数还是cv2.threshold()，但是需要多传入一个参数(flag):cv2.THRESH_OTSU。这时要把阈值设为0。然后算法会找到最优阈值，这个最优阈值就是返回值retVal。如果不使用Otsu二值化，返回的retVal值与设定的阈值相等
输入图像是一副带有噪声的图像。第一种方法，我们设127为全局阈值。第二种方法，我们直接使用Otsu二值化。第三种方法，我们首先使用5*5的高斯核出去噪音，然后再使用Otsu二值化。看看去除对结果的影响有多大
import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread()

#global thresholding
ret1,th1=cv2.threshold(img,127,255,cv2.THRESH_BINARY)

#Otsu's thresholding
ret2,th2=cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#Otsu's thresholding after Gaussian filtering
#(5,5)为高斯核的大小，0为标准差
blur=cv2.GaussianBlur(img,(5,5),0)
#阈值一定要设为0！
ret3,th3=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#plot all the images and their histograms
images=[img,0,th1,img,0,th2,blur,0,th3]
titles=['Orignal Noisy Image','Histogram','Global Thresholding(v=127)','Original Noisy Image','Histogram',"Otsu's Thresholding",'Gaussian filtered Image','Histogram',"Otsu's Thresholding"]

#这里使用了Pyplot中画直方图的方法，plt.hist，要注意他的参数是一维数组
#所以这里使用了(numpy)ravel方法，将多维数组转换为一维，也可以使用flatten方法
#ndarray.flat 1_D iterator over an array
#ndarray.flatten 1-D array copy of the elements of an array in row-major order
for i in range(3):
    plt.subplot(3,3,i*3+1),plt.imshow(images[i*3],'gray')
    plt.title(titles[i*3]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+2),plt.imshow(images[i*3].ravel(),256)
    plt.title(titles[i*3+1]),plt.xticks([]),plt.yticks([])
    plt.subplot(3,3,i*3+3),plt.imshow(images[i*3+2],'gray')
    plt.title(titles[i*3+2]),plt.xticks([]),plt.yticks([])
plt.show()


# Otsu's二值化是如何工作的

import cv2
import numpy as np

img=cv2.imread()
blur=cv2.GaussianBlur(img,(5,5),0)

#find normalized_histogram,and its cumulative distribution function
#计算归一化直方图
#CalcHist(image,accumulate=0,mask=NULL)
hist=cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm=hist.ravel()/hist.max()
Q=hist_norm.cumsum()

bins=np.arange(256)

fn_min=np.inf
thresh=-1

for i in range(1,256):
    p1,p2=np.hsplit(hist_norm,[i])#probabilities
    q1,q2=Q[i],Q[255]-Q[i] #cum sum of classes
    b1,b2=np.hsplit(bins,[i])# weights

    #finding means and variances
    m1,m2=np.sum(p1*b1)/q1,np.sum(p2*b2)/q2
    v1,v2=np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    #calculates the minization function
    fn=v1*q1+v2*q2
    if fn<fn_min:
        fn_min=fn
        thresh=i

#find tosu's theshold value with OpenCV function
ret,otsu=cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THREH_OTSU)
print thresh,ret


# 图像平滑
学习使用不同的低通滤波器对图像进行模糊
使用自定义的滤波器对图像进行卷积(2D卷积)

# 2D卷积
对2D图像实施低通滤波(LPF)，LPF帮助我们去除噪音，模糊图像的。HPF帮助我们找到图像的边缘
OpenCV提供的函数cv2.filter2D()可以对衣服图像进行卷积操作

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread('')
kernel=np.ones((5,5),np.float32)/25

dst=cv2.filter2D(img,-1,kernel)

plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]),plt.yticks([])

# 图像模糊(图像平滑)
使用低通滤波器可以达到图像模糊的目的。这对去除噪音很有帮助。其实就是去除图像中高频成分(噪音，边界)。所以边界也会模糊一点

## 平均
cv2.blur()和cv2.boxFilter()

import cv2
import numpy as np
from matplotlib import pyplot as plt
img=cv2.imread('')
blur-cv2.blur(img,(5,5))

## 高斯模糊
方框中心的值最大，其余方框根据距离中心元素的距离递减，构成一个高斯小山包。原来的平均数变成加权平均数，权就是方框里的值
cv2.GaussianBlur()
我们需要指定高斯核的宽和高(必须是奇数)。以及高斯函数沿X，Y方向的标准差
#高斯滤波可以有效的从图像中去除高斯噪音
可以使用cv2.getGaussianKernel()自己构建一个高斯核
blur=cv2.GaussianBlur(img,(5,5),0)

## 中值模糊
用卷积框对应像素的中值来替代中心像素的。这个滤波器经常用来去除椒盐噪声。卷积核的大小也应该是一个奇数
median=cv2.medianBlur(img,5)

## 双边滤波
cv2.bilateralFilter()能在保持边界清晰的情况下有效的去除噪声。这种操作与其他滤波器相比会比慢 
blur=cv2.bilateralFilter(img,9,75,75)

# 形态学转换
腐蚀、膨胀、开运算和闭运算
cv2.erode(),cv2.dilate(),cv2.morphologyEx()

原理：形态学操作是根据图像形状进行简单操作。一般情况下对二值化图像进行操作。需要收入两个参数，一个原始图像，第二个被称为结构化元素或核，它是用来决定操作的性质的。

## 腐蚀
这对于去除白色噪声很有用
import cv2
import numpy as np

img=cv2.imread()
kernel=np.ones((5,5),np.uint8)
erosion=cv2.erode(img,kernel,iterations=1)

## 膨胀
这个操作会增加图像中白色区域。一般在去噪声时会先用腐蚀再用膨胀。这时噪声已经被去除了，不会再回来了，但是前景还在并会增加。膨胀也可以用来连接两个分开的物体
dilation=cv2.dilate(img,kernel,iterations=1)

## 开运算
先进行腐蚀再进行膨胀就叫做开运算。它被用来去除噪声
cv2.morphologyEx()
opening=cv2.morphologyEx(img,cv2.MORPH_OPEN ,kernel)

## 闭运算
先膨胀在腐蚀。他经常被用来填充前景物体中的小洞，或者前景物体上的小黑点
closing=cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)

## 礼帽
原始图像与进行开运算之后得到的图像的差
tophat=cv2.morphologyEx(img,cv2.MORPH_TOPATH,kernel)

## 黑帽
进行闭运算后得到的图像与原始图像的差
blackhat=cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,kernel)

# 形态学操作之间的关系

## 结构化元素
我们使用Numpy构建了结构化元素，它是正方形的。但有时我们需要构建一个椭圆形/圆形的核。
cv2.getStructuringElement()。你只需要告诉他你需要的核的大小

// Rectangular Kernel
cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

// Elliptical Kernel
cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

//Cross-shaped Kernel
cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

# 图像梯度
图像梯度，图像边界
cv2.Sobel(),cv2.Schar(),cv2.Laplacian()等

原理:
梯度简单来说就是求导
OpenCV提供了三种不同的梯度滤波器，或者说是高通滤波器:Sobel,Scharr和Laplacian
Sobel,Scharr其实就是求一价或二阶导数
Scharr是对Sobel(使用小的卷积核求解梯度角度)的优化.Laplacian是求二阶导数

## Sobel算子和Scharr算子
Sobel算子是高斯平滑与为分操作的结合体，所以它的抗噪声好。你可以设定求导的方向(xorder或yorder)。还可以设定使用卷积核的大小(ksize)。如果ksize=-1，会使用3*3的Scharr滤波器，他的效果比3*3的Sobel滤波器好(而且速度相同，所以在使用3*3滤波器时应该尽量使用Scharr滤波器)

## Laplacian算子
拉普拉斯算子可以使用二阶导数的形式定义，可假设其离散实现类似于二阶Sobel导数。

import cv2
import numpy as np
from matplotlib import pyplot as plt

img=cv2.imread()

#cv.CV_64F 输出图像的深度(数据类型)，可以使用-1，与原图像保持一致np.uint8
laplacian=cv2.Lapacian(img,cv2.CV_64F)
sobelx=cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely=cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

plt.subplot(2,2,1),plt.imshow(img,cmap='gray')
plt.title('Original'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,2),plt.imshow(laplacian,cmap='gray')
plt.title('Laplacian'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,3),plt.imshow(sobelx,cmap='gray')
plt.title('Sobel X'),plt.xticks([]),plt.yticks([])
plt.subplot(2,2,4),plt.imshow(sobely,cmap='gray')
plt.title('Sobel Y'),plt.xticks([]),plt.yticks([])
plt.show()

#当我们可以通过参数-1来设定输出图像的深度(数据类型)与原图像保持一直，但是我们在代码中使用的确实cv2.CV_64F。这是为什么呢？想象以下一个从黑到比阿的边界的导数是整数，而一个从白到黑的边界点导数确实负数。如果原图像的深度是

# Canny边缘检测
OpenCV中的Canny检测
了解Canny边缘检测的概念
学习函数cv2.Canny()

#原理： Canny边缘检测是一种非常流行的边缘检测算法


## 噪声去除
由于边缘检测很容易受到噪声影响，第一部是使用5*5的高斯滤波器去除噪声

## 计算图像梯度
对平滑后的图像使用Sobel算子计算水平方向和竖直方向的一阶导数
根据得到的这两幅图(Gx和Gy)找到边界的梯度和方向
梯度一般总是与边界垂直
梯度方向被归为四类:垂直，水平和两个对角线

## 非极大值抑制
在获得梯度的方向和大小之后，应该对整幅图像做一个扫描，去除那些非边界上的点。对每个像素进行检查，看这个点的梯度是不是具有相同梯度方向的点中最大的

## 滞后阈值
现在要确定哪些边界才是真正的边界。这时我们需要设置两个阈值minVal和maxVal。当图像的灰度高于maxVal时被认为是真的边界，那些低于minVal的边界会被抛弃。如果介于两者之间的话，就要看这个点是否与某个被确定为真正的边界点相连，如果是就认为它也是边界点，如果不是就抛弃
在这一步一些小的噪声点也会被除去，因为我们假设边界都是一些长的线段

## OpenCV中的Canny边界检测
在OpenCV中只需要一个函数:cv2.Canny()
第一个参数是输入图形
第二个和第三个参数分别是minVal和maxVal。
第三个参数是设置用来计算图像梯度的Sobel卷积核的大小，默认值为3.最后一个参数是L2gradient，它可以用来设定求梯度大小的方程。如果设为True，就会使用我们上面提到的方程，否则使用|Gx^2|+|Gy^2|代替，默认值为False

import cv2
import numpy as np

img=cv2.imread()
edges=cv2.Canny(img,100,100)

plt.subplot(121),plt.imshow(img,cmap='gray')
plt.title('Original Image'),plt.xticks([]),plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap='gray')
plt.title('Edge Image'),plt.xticks([]),plt.yticks([])
plt.show()

# 图像金字塔
cv2.pyrUp()
cv2.pyrDown()

原理：我们要处理一幅具有固定分辨率的图像。但是有些情况下，我们需要对同一图像的不同分辨率的子图像进行处理。
有两类图像金字塔：高斯金字塔和拉普拉斯金字塔
高斯金字塔的顶部是通过将底部图像中的连续的行和列去除得到的。

函数cv2.pyrDown()从一个高分辨率大尺寸的图像向上构建一个金字塔(尺寸变小，分辨率降低)

img=cv2.imread('')
lower_reso=cv2.pyrDown(higher_reso)

函数cv2.pyrUp()从一个低分辨率小尺寸的图像向下构建一个金字塔(尺寸变大，但分辨率不会增加)
higher_reso2=cv2.pyrUp(lower_reso)

#higher_reso2和higher_reso是不同的，一旦使用cv2.pyrDown()，图像的分辨率就会降低，信息就会被丢失。

拉普拉斯金字塔的图像看起来就像边界图，其中很多像素都是0.他们经常被用在图像压缩中。

## 使用金字塔进行图像融合
图像金字塔的一个应用是图像融合

# OpenCV中的轮廓

## 初始轮廓
cv2.findContours()
cv2.drawContours()

cv2.findContours()有三个参数
第一个参数是输入图像
第二个参数是轮廓检索模式
第三个是轮廓层析结构
轮廓(第二个返回值)是一个Python列表，其中存储这图像中的所有轮廓，每一个轮廓都是一个Numpy数组，包含对象边界点(x,y)的坐标

cv2.drawContours()可以被用来绘制轮廓。他可以根据你提供的边界点绘制任何形状
第一个参数是原始图像
第二个参数是轮廓，一个Python列表
第三个参数是轮廓的索引(-1时绘制所有轮廓)。
轮廓的颜色和厚度等

import numpy as np
import cv2

im=cv2.imread()
imgray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(imgray,127,255,0)
image,contours,hierarchy=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

img=cv2.drawContour(img,contours,-1,(0,255,0),3)

img=cv2.drawContours(img,contours,3,(0,255,0),3)

# 轮廓的近似方法
cv2.findContours()第三个参数
轮廓是一个形状具有相同灰度值的边界。会存储形状边界上所有的(x,y)坐标.
cv2.CHAIN_APPROX_NONE,所有的边界点都会存储
cv2.CHAIN_ARPROX_SIMPLE 将轮廓上的沉余点都去掉，压缩轮廓，从而节省内存开支

# 轮廓特征
查找轮廓的不同特征，例如面积，周长，中心，边界框等

//矩
cv2.moments()会将计算得到的矩以一个字典的形式返回
img=cv2.imread()
ret,thresh=cv2.threshold(img,127,255,0)
contours,hierarch=cv2.findContours(thresh,1,2)

cnt=contours[0]
M=cv2.moments(cnt)
print M

计算对象的重心
cx=int(M['m10]/M['m00])
cy=int(M['m01']/M['m00'])

# 轮廓面积
cv2.contourArea()
area=cv2.contourArea(cnt)

# 轮廓周长(也被成为弧长)
cv2.arcLength()
第二个参数制定对象的形状是开还是闭(True)
perimeter=cv2.arcLength(cnt,True)

# 轮廓近似
将轮廓近似到另外一种由更少点组成的轮廓形状，新轮廓的点的数目由我们设定的准确度来确定
第二个参数叫做epsilon，从原始轮廓到近似轮廓的最大距离，是一个准确度参数
epsilon=0.1*cv2.arcLength(cnt,True)
approx=cv2.approxPolyDP(cnt,epsilon,True)

# 凸包
凸包与轮廓近似相似，但不同，有些情况下他们给出的结果是一样的
函数cv2.convexHull()可以用来检测一个曲线是否具有凸性缺陷，并能纠正缺陷。一般来说，凸性曲线总是凸出来的，至少是平的。如果有地方凹进取了就被叫做凸性缺陷

hull=cv2.convexHull(points[,clockwise[,returnPoints]])
points：传入的轮廓
hull:输出，通常不需要
clockwise方向标志。如果设置为True，输出的凸包是顺时针方向的
returnPoints默认值为True。它会返回凸包上点的坐标。如果设置为False，就会返回与凸包对应的轮廓上的点

hull=cv2.convexHull(cnt)
#如果向获得凸性缺陷，需要把returnPoints设置为False。

# 凸性检测
cv2.isContourConvex()可以用来检测一个曲线是不是凸的。他只能返回True或False
k=cv2.isContourConvex(cnt)

# 边界矩形

## 直边界矩形
一个直矩形(没有旋转的矩形)。它不会考虑对象是否旋转，所以边界矩形的面积不是最小的。可以使用cv2.boundingRect()查找得到
(x,y)为矩形左上角的坐标，(w,h)是矩形的宽和高
x,y,w,h=cv2.boundingRect(cnt)
img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

## 旋转的边界矩形
这个边界矩形是面皆最小的，因为它考虑了对象的旋转
cv2.minAreaRect()。返回的是一个Box2D结构
(x,y)为矩形左上角顶点(x,y)，(w,h)为矩形的宽和高,以及旋转角度.
通过函数cv2.boxPoints()获得矩形需要的4个角点
x,y,w,h=cv2.boundingRect(cnt)
img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

# 最小外接圆
函数cv2.minEnclosingCircle()寻找一个对象的最小外切圆
(x,y),radius=cv2.minEnclosingCircle(cnt)
center=(int(x),int(y))
radius=int(radius)
img=cv2.circle(img,center,radius,(0,255,0),2)

# 椭圆拟合
函数cv2.ellipse()，返回值其实就是旋转百年届矩形的内切圆
ellipse=cv2.fitEllipse(cnt)
img=cv2.ellipse(im,ellipse,(0,255,0),2)

# 直线拟合
根据一组点拟合出一条线，同样可以为图像的白色点拟合出一条线
rows,cols=img.shape[:2]
[vx,vy,x,y]=cv2.fitLine(cnt,cv2.DIST_L2,0,0.01,0.01)
lefty=int((-x(vy/vx))+y)
righty=int(((cols-x)*vy/vx)+y)
img=cv2.line(img,(cols-1,righty),(0,lefty),(0,255,0),2)

# 轮廓的性质
## 边界矩阵的宽高比
x,y,w,h=cv2.boundingRect(cnt)
aspect_ratio=float(w)/h

## extent
轮廓面积与边界矩形面积的比
area=cv2.contourArea(cnt)
x,y,w,h=cv2.boundingRect(cnt)
rect_area=w*h
extent=float(area)/rect_area

## solidity
轮廓面积与凸包面积的比
area-cv2.contourArea(cnt)
hull=cv2.convexHull(cnt)
hull_area=cv2.contourArea(hull)
solidity=float(area)/hull_area

## Equivalent Diameter
与轮廓面积相等的圆形的直径
area=cv2.contourArea(cnt)
equi_diameter=np.sqrt(4*area/np.pi)

## 方向
对象的方向，下面的方法还会返回长轴和短轴
(x,y),(MA,ma),angle=cv2.fitEllipse(cnt)

## 掩膜和像素点
有时我们需要构成所有对象的像素点
mask=np.zeros(imgray.shape,np.uint8)

#这里一定要使用参数-1,绘制填充的轮廓
cv2.drawContours(mask,[cnt],0,255,-1)

pexelpints=np.transpose(np.nonzero(mask))

## 最大值和最小值以及他们的位置
使用掩膜图像得到这些参数
min_val,max_val,min_loc,max_loc=cv2.minMaxLoc(imgray,mask=mask)

## 平均颜色以及平均灰度
使用掩膜求一个对象的平均颜色或平均灰度
mean_val=cv2.mean(im,mask=mask)

## 极点
leftmost=tuple(cnt[cnt[:,:,0].argmin()][0])
rightmost=tuple(cnt[cnt[:,:,0].argmax()][0])
topmost=tuple(cnt[cnt[:,:,1].argmin()][0])
bottommost=tuple(cnt[cnt[:,:,1].argmax()][0])

# 轮廓:更多参数
凸缺陷，以及如何找到凸缺陷
找到某一点的一个多边形的最短距离
不同形状的匹配

## 凸缺陷
对象上的任何凹陷都被称为凸缺陷
cv2.convexityDefect()可以帮助我们找到凸缺陷
hull=cv2.convexHull(cnt,returnPoints=False)
defects=cv2.convexityDefects(cnt,hull)
#如果要查找凸缺陷，在使用函数cv2.convexHull找凸包时，参数returnPoints一定要是False

它会返回一个数组，其中每一行包含的值是[起点，终点，最远的点，到最远点的近似距离]

import cv2
import numpy as np

img=cv2.imread('')
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(img_gray,127,255,0)
contours,hierarchy=cv2.findContours(thresh,2,1)

hull=cv2.convexHull(cnt,returnPoints=False)
defects=cv2.convexityDefects(cnt,hull)

for i in range(defects.shape[0]):
    s,e,f,d=defects[i,0]
    start=tuple(cnt[s][0])
    end=tuple(cnt[e][0])
    far=tuple(cnt[f][0])
    cv2.line(img,start,end,[0,255,0],2)
    cv2.circle(img,far,5.[0,0,255],-1)

cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

## Point Polygon Test
求解图像中的一个点到一个对象轮廓的最短距离
如果点在轮廓的外部，返回值为-。如果在轮廓上，返回值为0。如果在轮廓内部，返回值为+
dist=cv2.pointPolygonTest(cnt,(50,50),True)
第三个参数为measureDist。如果设置为True，就会计算最短距离。如果是False,只会判断这个点与轮廓之间的位置关系

## 形状匹配
函数cv2.matchShape()可以帮助我们比较两个形状或轮廓的相似度，如果返回值越小，匹配越好。它是根据Hu距来计算的

import cv2
import numpy as np

img1=cv2.imread('',0)
img2=cv2.imread(''，0)

ret,thresh1=cv2.threshold(img1,127,255,0)
ret,thresh2=cv2.threshold(img2,127,255,0)
contours,hierarchy=cv2.findContours(thresh1,2,1)
cnt1=contours[0]
contours,hierarchy=cv2.findContours(thresh2,2,1)
cnt2=contours[0]

ret=cv2.matchShapes(cnt1,cnt2,1,0.0)
print ret

# 轮廓的层次结构
原理：cv2.findContours来查找轮廓，我们需要传入一个参数:轮廓提取模式(Contour_Retrieval_mode)。我们总是把它设置为cv2.RETR_LIST或者是cv2.RETR_TREE,效果还可以
同时，我们得到的结果包含3个数组，第一个图像，第二个是轮廓，第三个是层次结构。

## OpenCV中的层次结构
OpenCV使用一个含有四个元素的数组表示[Next,Previous,First_Child,Parent]
Next表示同一级组织结构中的下一个轮廓
Previous表示同一级结构中的前一个轮廓
First_Child表示它的第一个子轮廓
Parent表示它的父轮廓


## OpenCV中的检索模式
cv2.RETR_LIST,CV2.RETR_TREE,cv2.RETR_CCOMP,cv2.RETR_EXTERNAL
RETR_LIST 属于同一级组织轮廓
RETR_EXTERNAL 只会返回最外边的轮廓，所有的子轮廓都会被忽略掉
RETR_CCOMP 返回所有的轮廓并将轮廓分为两级组织结构
RETR_TREE 返回所有轮廓，并且创建一个完整的组织结构列表。

# 直方图
## 直方图的计算，绘制与分析
使用OpenCV或Numpy函数计算直方图
使用OpenCV或Matplotlib函数绘制直方图
cv2.calcHist(),np.histogram()

## 原理:通过直方图你可以对整幅图像的灰度分布有一个整体的了解。直方图的x轴是灰度值(0到255),y轴是图片中具有同一个灰度值点的数目
通过直方图我们可以对图像的对比度，亮度，灰度分布等有一个直观的认识。几乎所有的图像处理软件都提供了直方图分析功能。
#要知道，直方图是根据灰度图像绘制的，而不是彩色图像。直方图的左边区域像是暗一点的像素数量，右侧显示了亮一点的像素数量

## 统计直方图
bins：上面的直方图显示了每个灰度值对应的像素值。如果像素值为0-255，你就需要256个数来显示上面的直方图。OpenCV的文档中用histSize表示BINS  
dims:表示我们收集数据的参数数目。在本例子中，我们对收集到的数据只考虑一件事:灰度值。dims=1
range:统计的灰度值的范围:一般来说为[0,256]，也就是说所有的灰度值

## 使用OpenCV统计直方图
函数cv2.calcHist可以帮助我们统计一幅图像的直方图
cv2.calcHist(images,channels,mask,histSize,ranges[,hist[,accumulate]])
images: 原图像(图像个是为uint8或float32)。当传入函数时应该用中括号[]括起来，例如[img]
