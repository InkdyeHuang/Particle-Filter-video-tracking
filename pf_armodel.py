from cv2 import cv2
import copy
import numpy as np
from numpy.random import *
from skimage.measure import compare_ssim as ssim
from functools import cmp_to_key
import multiprocessing
from time import sleep

drawing = False
ix,iy,w,h= -1, -1,-1,-1
cropped = None
firstFrame = None
the_firstFrame = None
is_cropped = False
particle_num = 100
image_w,image_h = -1,-1

#高斯噪音的标准差
TRANS_X_STD = 1.0
TRANS_Y_STD = 0.5
TRANS_S_STD = 0.001

#二阶动态回归模型参数
A1  = 2.0
A2 = -1.0
B0  = 1.0000

class Particle(object):
    def __init__(self,_x = 0,_y = 0,_s = 0,_w = 0,_h = 0):
        self.x = _x
        self.y = _y
        self.s = _s
        self.xp = _x
        self.yp = _y
        self.sp = _s
        self.x0 = _x
        self.y0 = _y
        self.width = _w
        self.height = _h
        self.weight = 1

    #使用二阶动态回归来更新粒子状态
    def transition(self,frame,rng = 0):
        up_x = A1 * (self.x - self.x0) + A2 * (self.xp - self.x0) + B0 * add_guassnoise(rng,TRANS_X_STD) + self.x0
        up_x = max(0.0, min(image_w-1.0,up_x))
        up_y = A1 * (self.y - self.y0) +  A2 * (self.yp - self.y0) + B0 * add_guassnoise(rng,TRANS_Y_STD) + self.y0
        up_y = max(0.0, min(image_h-1.0,up_y))
        up_s = A1 * (self.s - 1.0) + A2 * (self.sp - 1.0) + B0 * add_guassnoise(rng,TRANS_S_STD) + 1.0
        # print(up_s,self.s)
        up_s = max(0.1,up_s)
        self.xp = self.x
        self.yp = self.y
        self.sp = self.s
        self.x = up_x
        self.y = up_y
        self.s = up_s
        y0 = max(0,int(self.y - h * self.s * 0.5))
        y1 = max(0,int(self.y + h * self.s * 0.5))
        x0 = max(0,int(self.x - w * self.s * 0.5))
        x1 = max(0,int(self.x + w * self.s * 0.5))
        # print(self.x,self.y)
        #print(y0,y1,x0,x1)
        noisy_sub = frame[y0: y1,x0:x1]
        noisy_sub = cv2.resize(noisy_sub,(cropped.shape[1],cropped.shape[0]),interpolation=cv2.INTER_CUBIC)
        self.weight =max(0.0, ssim(cropped,noisy_sub,multichannel=True))
        #print(self.weight)
def choose_frame(event, x, y, flags, param):
    global ix, iy,w,h,drawing,cropped,is_cropped
    if event == cv2.EVENT_LBUTTONDOWN:
        print('left button down')
        drawing = True
        ix, iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        print('mouse move')
        if drawing == True:
            cv2.rectangle(firstFrame, (ix, iy), (x,y), (0,255,0), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        print('left button up')
        drawing = False
        is_cropped = True
        cropped = the_firstFrame[iy:y,ix:x]
        w,h = x - ix,y - iy
        ix = ix + w * 0.5
        iy = iy + h * 0.5
        cv2.destroyAllWindows()
        cv2.imshow("cropped",cropped)
        cv2.waitKey(10)
def normal_weight(particle_list):
    Sum = 0
    for p in particle_list:
        Sum += p.weight
    for i in range(particle_num):
        particle_list[i].weight /= Sum
    return particle_list
def add_noise(mat):
    mat = mat + np.random.normal(0,3,mat.shape[0] * mat.shape[1]).reshape(mat.shape)
    for i in range(particle_num):
        mat[i][0] %= image_w
        mat[i][1] %= image_h
    return mat
def add_guassnoise(rng,mem):
    return np.random.normal(rng,mem)
def resample(particle_list):
    new_particle_list = []
    k = 0
    for i in range(particle_num):
       np = int(round(particle_list[i].weight * particle_num))
       for j in range(np + 1):
            if k < particle_num:
                new_particle_list.append(copy.deepcopy(particle_list[i]))
                k += 1
    return new_particle_list
def initial_particle():
    particle_list = []
    for i in range(particle_num):
        p = Particle(ix,iy,1,w,h)
        particle_list.append(p)
    particle_list = normal_weight(particle_list)
    particle_list = resample(particle_list)
    return particle_list
def particlefilter(particle_list,frame):
    #print(len(particle_list))
    for i in range(particle_num):
        particle_list[i].transition(frame)
    particle_list = normal_weight(particle_list)
    particle_list = resample(particle_list)
    frame = show_predict(particle_list,frame)
    return particle_list,frame
def particle_cmp(p1,p2):
    if p1.weight < p2.weight:
        return 1
    elif p1.weight > p2.weight:
        return  -1
    return 0
# def show_predict(particle_list,frame):
#     sx,sy,ss = 0,0,0
#     for p in particle_list:
#         sx += p.x * p.weight
#         sy += p.y * p.weight
#         ss += p.s * p.weight
#     ix = int(sx - w * ss  * 0.5)
#     iy = int(sy - h * ss * 0.5)
#     sx =  int(sx + w * ss  * 0.5)
#     sy = int(sy + h * ss * 0.5)
#     cv2.rectangle(frame, (ix, iy), (sx,sy), (0,255,0), 1)
#     return frame
def show_predict(particle_list,frame):
    sx,sy,ss = 0,0,0
    particle_list.sort(key = cmp_to_key(lambda a,b : (b.weight - a.weight)))
    index = 0
    for p in particle_list[:2]:
        index += 1
        #print("index = ",index,p.weight)
        ix = int(p.x - w * p.s  * 0.5)
        iy = int(p.y - h * p.s * 0.5)
        sx =  int(p.x + w * p.s  * 0.5)
        sy = int(p.y + h * p.s * 0.5)
        cv2.rectangle(frame, (ix, iy), (sx,sy), (0,255,0), 1)
    return frame
if __name__ == "__main__":
    capture = cv2.VideoCapture("video/hockey.avi")
    is_destroy = False
    particle_list = []
    if capture.isOpened():
        while(True):
            ret,prev = capture.read()
            if ret == True:
                if is_cropped == False:
                    firstFrame = copy.deepcopy(prev)
                    the_firstFrame = copy.deepcopy(prev)
                    image_w,image_h = prev.shape[1],prev.shape[0]
                while(is_cropped == False):
                    cv2.namedWindow("choose_image",flags = 0)
                    cv2.resizeWindow('choose_image', 1080, 800) 
                    cv2.setMouseCallback('choose_image', choose_frame)
                    cv2.imshow('choose_image',firstFrame)
                    cv2.waitKey(10)&0xff
                if is_destroy == False :
                    cv2.destroyAllWindows()
                    is_destroy = True
                    particle_list = initial_particle()
                    cv2.rectangle(prev, (int(ix - w * 0.5), int(iy - 0.5 * h)), (int(ix + w * 0.5), int(iy + 0.5 * h)), (0,255,0), 1)
                    cv2.namedWindow('video', flags=0)  
                    cv2.resizeWindow('video', 1080, 800) 
                    cv2.imshow('video',prev)
                else:
                    particle_list,prev = particlefilter(particle_list,copy.deepcopy(prev))
                    cv2.imshow('video',prev)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()