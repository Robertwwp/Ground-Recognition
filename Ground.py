
'''
Opencv ground area detection
'''

from __future__ import print_function

import numpy as np
import cv2

cam = cv2.VideoCapture(1)

#return the values of standard deviations of brightness in different regions
def std_bri(img,step,h,w,y,x):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bri=np.zeros(shape=(len(y),len(x)))
    #bri_flag=np.zeros(shape=(len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            bri[i][j]=np.std(gray[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            #if bri[i][j]>20:
                #bri_flag[i][j]=1
                #img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,255,0),3)
    return bri

#return the values of standard deviations and average of color in different regions
def color(img,step,h,w,y,x):
    b,g,r=cv2.split(img)
    color_b=np.zeros(shape=(len(y),len(x)))
    color_g=np.zeros(shape=(len(y),len(x)))
    color_r=np.zeros(shape=(len(y),len(x)))
    color_std=np.zeros(shape=(len(y),len(x)))
    color_avg=np.zeros(shape=(len(y),len(x)))
    #color_flag=np.zeros(shape=(len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            color_b[i][j]=np.std(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_g[i][j]=np.std(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_r[i][j]=np.std(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_std[i][j]=color_b[i][j]+color_g[i][j]+color_r[i][j]
            color_b[i][j]=np.mean(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_g[i][j]=np.mean(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_r[i][j]=np.mean(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_avg[i][j]=color_b[i][j]+color_g[i][j]+color_r[i][j]
            #if color_b[i][j]+color_g[i][j]+color_r[i][j]>100:
                #color_flag[i][j]=1
                #img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    return color_std,color_avg

#return the optical flow of different regions
def OPflow(previmg):
    prevgray = cv2.cvtColor(previmg, cv2.COLOR_BGR2GRAY)
    cap=input('press 1 if the camera is oriented properly')
    if cap==1:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

'''homography induced similarity'''
def Homo():



i=0
while i<1:

    ret, img = cam.read()
    step=32;
    h,w=img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)

    bri=std_bri(img,step,h,w,y,x)
    color_std,color_avg=color(img,step,h,w,y,x)
    flow=OPflow(img)

    #pixels in the lower middle part of the image represent the ground
    g_cri_bri=(bri[len(y)-1][len(x)/2-1]+bri[len(y)-1][len(x)/2-2]+bri[len(y)-1][len(x)/2]+
               bri[len(y)-2][len(x)/2-1]+bri[len(y)-2][len(x)/2-2]+bri[len(y)-2][len(x)/2])/6
    g_cri_cstd=(color_std[len(y)-1][len(x)/2-1]+color_std[len(y)-1][len(x)/2-2]+color_std[len(y)-1][len(x)/2]+
                color_std[len(y)-2][len(x)/2-1]+color_std[len(y)-2][len(x)/2-2]+color_std[len(y)-2][len(x)/2])/6
    g_cri_cavg=(color_avg[len(y)-1][len(x)/2-1]+color_avg[len(y)-1][len(x)/2-2]+color_avg[len(y)-1][len(x)/2]+
                color_avg[len(y)-2][len(x)/2-1]+color_avg[len(y)-2][len(x)/2-2]+color_avg[len(y)-2][len(x)/2])/6
    g_cri_flow=(flow[len(y)-1][len(x)/2-1][0]+flow[len(y)-1][len(x)/2-2][0]+flow[len(y)-1][len(x)/2][0]+
                flow[len(y)-2][len(x)/2-1][0]+flow[len(y)-2][len(x)/2-2][0]+flow[len(y)-2][len(x)/2][0])/6
    print(g_cri_flow)

    #compare, decide and draw the ground regions
    for i in range(len(y)):
        for j in range(len(x)):
            '''weighted method...'''
            if abs(bri[i][j]-g_cri_bri)<40 and abs(color_std[i][j]-g_cri_cstd)<30 and abs(color_avg[i][j]-g_cri_cavg)<50 and abs(flow[i][j][0]-g_cri_flow)<0.8:
            #if color_flag[i][j]==1 and bri_flag[i][j]==1:
                img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    cv2.imshow('image',img)
    i=i+1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
