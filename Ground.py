
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
        cam = cv2.VideoCapture(1)
        ret, img = cam.read()
        cv2.imshow('op',img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prevgray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        cam.release()
    return flow

#find the correspondening point
def find_corr(y,x,prev,img):
    bp,gp,rp = cv2.split(prev)
    b,g,r = cv2.split(img)

    corr=np.zeros((5,2),dtype=np.float32)
    for index in range(len(y)):
        pix_b=float(bp[y[index]][x[index]])
        pix_g=float(gp[y[index]][x[index]])
        pix_r=float(rp[y[index]][x[index]])

        corr_sum_i=0
        corr_sum_j=0
        corr_count=0
        for i in range(16):
            for j in range(16):
                #turn the camera to the right
                a=abs(float(b[y[index]+i][x[index]-j])-pix_b)+abs(float(g[y[index]+i][x[index]-j])-pix_g)+abs(float(r[y[index]+i][x[index]-j])-pix_r)
                if a<5:
                        corr_sum_i=corr_sum_i+i
                        corr_sum_j=corr_sum_j+j
                        corr_count=corr_count+1
        if corr_count!=0:
            corr[index][0]=y[index]+int(corr_sum_i/corr_count)
            corr[index][1]=x[index]-int(corr_sum_j/corr_count)
            cv2.circle(img, (corr[index][1], corr[index][0]), 1, (0, 255, 0), 2)
        else:
            '''need to be dealt with'''
            print('failed to find a match')
            corr[index]=[y[index],x[index]]
    return corr,img


'''homography induced similarity'''
'''find corespondences/ for each region? for edges? for the whole image?'''
def Homography(y,x,step,prev):
    #change to robot's camera motion later
    cap=input('press 1 if the camera is oriented properly')
    if cap==1:
        cam = cv2.VideoCapture(1)
        ret, img = cam.read()

        Homo=np.zeros((len(y),len(x),9),dtype=np.float32)
        '''missing the outer parts'''
        for i in range(1,len(y)-1):
            for j in range(1,len(x)-1):
                x_c=[x[j]-step/2,x[j],x[j],x[j],x[j]+step/2]
                y_c=[y[i],y[i]-step/2,y[i],y[i]+step/2,y[i]]
                src_pts=np.zeros((5,2),dtype=np.float32)
                for a in range(5):
                    src_pts[a][0]=y_c[a]
                    src_pts[a][1]=x_c[a]
                dst_pts,img=find_corr(y_c,x_c,prev,img)
                H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                Homo[i][j]=np.reshape(H,9)

    return Homo,img




i=0
while i<1:

    ret, img = cam.read()
    step=32;
    h,w=img.shape[:2]
    y=np.linspace(step/2,h-step/2,15,dtype=int)
    x=np.linspace(step/2,w-step/2,20,dtype=int)

    bri=std_bri(img,step,h,w,y,x)
    color_std,color_avg=color(img,step,h,w,y,x)
    cam.release()
    Homo,img=Homography(y,x,step,img)
    print(Homo)
    '''need to test the match and find implement the homography criterion'''
    #flow=OPflow(img)

    #pixels in the lower middle part of the image represent the ground
    g_cri_bri=(bri[len(y)-1][len(x)/2-1]+bri[len(y)-1][len(x)/2-2]+bri[len(y)-1][len(x)/2]+
               bri[len(y)-2][len(x)/2-1]+bri[len(y)-2][len(x)/2-2]+bri[len(y)-2][len(x)/2])/6
    g_cri_cstd=(color_std[len(y)-1][len(x)/2-1]+color_std[len(y)-1][len(x)/2-2]+color_std[len(y)-1][len(x)/2]+
                color_std[len(y)-2][len(x)/2-1]+color_std[len(y)-2][len(x)/2-2]+color_std[len(y)-2][len(x)/2])/6
    g_cri_cavg=(color_avg[len(y)-1][len(x)/2-1]+color_avg[len(y)-1][len(x)/2-2]+color_avg[len(y)-1][len(x)/2]+
                color_avg[len(y)-2][len(x)/2-1]+color_avg[len(y)-2][len(x)/2-2]+color_avg[len(y)-2][len(x)/2])/6
    '''g_cri_flow=(flow[len(y)-1][len(x)/2-1][0]+flow[len(y)-1][len(x)/2-2][0]+flow[len(y)-1][len(x)/2][0]+
                flow[len(y)-2][len(x)/2-1][0]+flow[len(y)-2][len(x)/2-2][0]+flow[len(y)-2][len(x)/2][0])/6
    print(g_cri_flow)'''

    #compare, decide and draw the ground regions
    for i in range(len(y)):
        for j in range(len(x)):
            '''weighted method...optical flow has less impact for the pixels far away,can be used just for simple test'''
            if abs(bri[i][j]-g_cri_bri) + abs(color_std[i][j]-g_cri_cstd) +abs(color_avg[i][j]-g_cri_cavg)<80:
                #and abs(flow[i][j][0]-g_cri_flow)<0.8:
            #if color_flag[i][j]==1 and bri_flag[i][j]==1:
                img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    cv2.imshow('image',img)
    i=i+1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

#cam.release()
cv2.destroyAllWindows()
