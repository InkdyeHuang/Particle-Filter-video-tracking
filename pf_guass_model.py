from cv2 import cv2
import copy
import numpy as np
from numpy.random import *
from skimage.measure import compare_ssim as ssim

drawing = False
ix,iy,w,h= -1, -1,-1,-1
cropped = None
firstFrame = None
the_firstFrame = None
is_cropped = False
particle_num = 100
image_w,image_h = -1,-1
class Particle(object):
    def __init__(self,_x = 0,_y = 0,_s = 0):
        self.x = _x
        self.y = _y
        self.s = _s
        self.xp = 0
        self.yp = 0
        self.sp = 0
        self.x0 = 0
        self.y0 = 0
        self.width = 0
        self.height = 0
        self.weight = 0

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
        w,h = x - ix,y - iy
        drawing = False
        cropped = the_firstFrame[iy:y,ix:x]
        is_cropped = True
        cv2.destroyAllWindows()
        cv2.imshow("cropped",cropped)
        cv2.waitKey(10)
def add_noise(mat):
    mat = mat + np.random.normal(0,3,mat.shape[0] * mat.shape[1]).reshape(mat.shape)
    for i in range(particle_num):
        mat[i][0] %= image_w
        mat[i][1] %= image_h
    return mat

def resample(weights):               
    weights = sorted(weights, reverse = True)
    n = len(weights)
    indices = []
    C = [0.] + [sum(weights[:i+1]) for i in range(n)]
    u0,j = random(),0
    for u in [(u0 + i) / n  for i in range(n)]:
        while u > C[j]:
            j+=1
        indices.append(j - 1)
    return indices

def initial_particle():
    weights =[]
    x = np.array([[ix,iy] for i in range(particle_num)])
    x = add_noise(x)
    for now_x in x:
        lx,ly = int(now_x[0]),int(now_x[1])
        if lx + w  > image_w or ly + h > image_h:
            noisy_sub = the_firstFrame[ly : ly + h,lx: lx + w]
            noisy_sub = cv2.resize(noisy_sub, (w,h), interpolation=cv2.INTER_CUBIC)
            weights.append(ssim(cropped,noisy_sub,multichannel=True))
        else:
            noisy_sub = the_firstFrame[ly : ly + h,lx: lx + w]
            weights.append(ssim(cropped,noisy_sub,multichannel=True))
    weights = weights / sum(weights)
    indice = resample(weights)
    y = np.array([x[i] for i in indice])
    x = y

    return x
def particlefilter(x,frame):
    weights = []
    x = add_noise(x)
    for now_x in x:
        lx,ly = int(now_x[0]),int(now_x[1])
        if lx + w  > image_w or ly + h > image_h:
            noisy_sub = the_firstFrame[ly : ly + h,lx: lx + w]
            noisy_sub = cv2.resize(noisy_sub, (w,h), interpolation=cv2.INTER_CUBIC)
            weights.append(ssim(cropped,noisy_sub,multichannel=True))
        else:
            noisy_sub = frame[ly : ly + h,lx: lx + w]
            weights.append(ssim(cropped,noisy_sub,multichannel=True))
    weights = weights / sum(weights)
    weights = weights / sum(weights)
    indice = resample(weights)
    y = np.array([x[i] for i in indice])
    x = y
    return weights,x,frame

if __name__ == "__main__":
    capture = cv2.VideoCapture("video/hockey.avi")
    is_destroy = False
    x = []
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
                    cv2.resizeWindow('choose_image', 1080, 600) 
                    cv2.setMouseCallback('choose_image', choose_frame)
                    cv2.imshow('choose_image',firstFrame)
                    cv2.waitKey(10)&0xff
                if is_destroy == False :
                    cv2.waitKey(1000)
                    cv2.destroyAllWindows()
                    is_destroy = True
                    x = initial_particle()
                    cv2.rectangle(prev, (ix, iy), (ix + w,iy + h), (0,255,0), 1)
                    cv2.namedWindow('video', flags=0)  
                    cv2.resizeWindow('video', 1080, 600) 
                    cv2.imshow('video',prev)
                else:
                    weights,x,prev = particlefilter(x,prev)
                    sx,sy = 0,0
                    for i,now_x in enumerate(x):
                        sx += now_x[0] * weights[i]
                        sy += now_x[1] * weights[i]
                    sx = int(sx)
                    sy = int(sy)
                    print(sx,sy)
                    cv2.rectangle(prev, (sx, sy), (sx + w,sy + h), (0,255,0), 1)
                    cv2.imshow('video',prev)
            else:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    capture.release()
    cv2.destroyAllWindows()