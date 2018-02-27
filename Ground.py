
'''
Opencv ground area detection
'''

from __future__ import print_function

import numpy as np
import cv2

cam = cv2.VideoCapture(1)

#return the values of standard deviations of brightness in different regions
#maybe add average
def bri(img,step,h,w,y,x):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bri_std=np.zeros(shape=(len(y),len(x)))
    bri_mean=np.zeros(shape=(len(y),len(x)))
    #bri_flag=np.zeros(shape=(len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            bri_std[i][j]=np.std(gray[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            bri_mean[i][j]=np.mean(gray[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            #if bri[i][j]>20:
                #bri_flag[i][j]=1
                #img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,255,0),3)
    return bri_std,bri_mean

#return the values of standard deviations and average of color in different regions
def color(img,step,h,w,y,x):
    b,g,r=cv2.split(img)
    color_bstd=np.zeros(shape=(len(y),len(x)))
    color_gstd=np.zeros(shape=(len(y),len(x)))
    color_rstd=np.zeros(shape=(len(y),len(x)))
    color_bavg=np.zeros(shape=(len(y),len(x)))
    color_gavg=np.zeros(shape=(len(y),len(x)))
    color_ravg=np.zeros(shape=(len(y),len(x)))

    #color_flag=np.zeros(shape=(len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            color_bstd[i][j]=np.std(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_gstd[i][j]=np.std(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_rstd[i][j]=np.std(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_bavg[i][j]=np.mean(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_gavg[i][j]=np.mean(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_ravg[i][j]=np.mean(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            #if color_b[i][j]+color_g[i][j]+color_r[i][j]>100:
                #color_flag[i][j]=1
                #img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    return color_bstd,color_gstd,color_rstd,color_bavg,color_gavg,color_ravg

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

    corr=np.zeros((len(y),2),dtype=np.float32)
    f=0
    for index in range(len(y)):
        pix_b_mat=bp[y[index]-1:y[index]+1,x[index]-1:x[index]+1]
        pix_g_mat=gp[y[index]-1:y[index]+1,x[index]-1:x[index]+1]
        pix_r_mat=rp[y[index]-1:y[index]+1,x[index]-1:x[index]+1]

        corr_sum_i=0
        corr_sum_j=0
        corr_count=0

        '''maybe larger correspondence check range'''
        for i in range(16):
            for j in range(16):
                #turn the camera to the right
                a=np.sum(abs(np.add(b[y[index]+i-1:y[index]+i+1,x[index]-j-1:x[index]-j+1],-pix_b_mat))
                 +abs(np.add(g[y[index]+i-1:y[index]+i+1,x[index]-j-1:x[index]-j+1],-pix_g_mat))
                 +abs(np.add(r[y[index]+i-1:y[index]+i+1,x[index]-j-1:x[index]-j+1],-pix_r_mat)))
                if a<30:
                        corr_sum_i=corr_sum_i+i
                        corr_sum_j=corr_sum_j+j
                        corr_count=corr_count+1
        if corr_count!=0:
            corr[index][0]=y[index]+int(corr_sum_i/corr_count)
            corr[index][1]=x[index]-int(corr_sum_j/corr_count)
            cv2.circle(img, (corr[index][1], corr[index][0]), 1, (0, 255, 0), 2)
            f=f+1
    if f>=4:
        f=1
    else:
        f=0
    return corr,img,f


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
                x_c,y_c=np.mgrid[x[j]-step/2:x[j]+step/2:step/4, y[i]-step/2:y[i]+step/2:step/4].reshape(2,-1).astype(int)
                src_pts=np.zeros((len(x_c),2),dtype=np.float32)
                for a in range(len(x_c)):
                    src_pts[a][0]=y_c[a]
                    src_pts[a][1]=x_c[a]
                dst_pts,img,f=find_corr(y_c,x_c,prev,img)
                if f==1:
                    H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                    #print(H)
                    if H is not None:
                        Homo[i][j]=np.reshape(H,9)

    return Homo,img

i=0
while i<1:

    ret, img = cam.read()
    step=16;
    h,w=img.shape[:2]
    y=np.linspace(step/2,h-step/2,int((h-step)/step),dtype=int)
    x=np.linspace(step/2,w-step/2,int((w-step)/step),dtype=int)

    bri_std,bri_mean=bri(img,step,h,w,y,x)
    color_bstd,color_gstd,color_rstd,color_bavg,color_gavg,color_ravg=color(img,step,h,w,y,x)
    cam.release()
    Homo,img2=Homography(y,x,step,img)
    #print(Homo)
    '''need to test the match and find implement the homography criterion'''
    #flow=OPflow(img)

    #pixels in the lower middle part of the image represent the ground
    #print(bri)
    g_cri_bstd=np.sum(bri_std[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_bavg=np.sum(bri_mean[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cbstd=np.sum(color_bstd[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cgstd=np.sum(color_gstd[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_crstd=np.sum(color_rstd[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cbavg=np.sum(color_bavg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cgavg=np.sum(color_gavg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cravg=np.sum(color_ravg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9

    '''get more representitive ground homography'''
    g_cri_homo=np.zeros(9)
    count=0
    for m in range(-5,-2):
        for n in range(-1,2):
            if np.sum(Homo[len(y)+m][len(x)/2+n])!=0:
                g_cri_homo=sum(g_cri_homo,Homo[len(y)+m][len(x)/2+n])
                count=count+1
    print(count)
    g_cri_homo=g_cri_homo/count
    print(g_cri_homo)

    #g_cri_homo=sum(Homo[len(y)-5:len(y)-1,len(x)/2-3:len(x)/2+1])/16
    #print(Homo[len(y)-5:len(y)-1,len(x)/2-3:len(x)/2+1])
    '''g_cri_flow=(flow[len(y)-1][len(x)/2-1][0]+flow[len(y)-1][len(x)/2-2][0]+flow[len(y)-1][len(x)/2][0]+
                flow[len(y)-2][len(x)/2-1][0]+flow[len(y)-2][len(x)/2-2][0]+flow[len(y)-2][len(x)/2][0])/6
    print(g_cri_flow)'''

    #compare, decide and draw the ground regions
    for i in range(len(y)):
        for j in range(len(x)):
            '''find bettter method to check homography similarity'''
            if np.sum(Homo[i][j])!=0 and np.sum(np.add(Homo[i][j],-g_cri_homo))<0.01:
                img2 = cv2.rectangle(img2,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
            '''weighted method...optical flow has less impact for the pixels far away,can be used just for simple test'''
            if (abs(bri_std[i][j]-g_cri_bstd)<50 and abs(bri_mean[i][j]-g_cri_bavg)<50
                and abs(color_bstd[i][j]-g_cri_cbstd)<15 and abs(color_bavg[i][j]-g_cri_cbavg)<20
                and abs(color_gstd[i][j]-g_cri_cgstd)<15 and abs(color_gavg[i][j]-g_cri_cgavg)<20
                and abs(color_rstd[i][j]-g_cri_crstd)<15 and abs(color_ravg[i][j]-g_cri_cravg)<20):
                #and abs(flow[i][j][0]-g_cri_flow)<0.8:
            #if color_flag[i][j]==1 and bri_flag[i][j]==1:
                img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    cv2.imshow('image1',img)
    cv2.imshow('image2',img2)

    i=i+1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

#cam.release()
cv2.destroyAllWindows()
