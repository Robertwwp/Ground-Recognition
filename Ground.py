
'''
Opencv ground area detection
'''

from __future__ import print_function
import numpy as np
import cv2
import time

#np.set_printoptions(threshold=np.inf)

H_R = np.array([[ 1.10587804e+00, -6.43249198e-02, -1.84515951e+02],
                [ 2.89657049e-02,  1.06138065e+00, -2.79427738e+01],
                [ 1.40568837e-04,  1.54480500e-05,  1.00000000e+00]],dtype=np.float32)

H_L = np.array([[ 8.41595347e-01,  7.80677915e-02,  2.81009575e+02],
                [-3.83048857e-02,  9.50095266e-01,  3.37983703e+01],
                [-2.01290838e-04, -4.32323499e-05,  1.00000000e+00]],dtype=np.float32)

H_B = np.array([[ 1.03942196e+00,  1.42230512e-01, -2.25065316e+01],
                [ 5.13076092e-04,  1.10165364e+00, -1.02445581e+01],
                [-2.50237564e-05,  4.08002281e-04,  1.00000000e+00]],dtype=np.float32)

H_BB = np.array([[ 9.98412114e-01, -1.91174655e-01, -2.33504844e+01],
                [ 1.06039618e-02,  9.06643631e-01, -2.01838585e+00],
                [ 6.56247635e-05, -3.90300252e-04,  1.00000000e+00]],dtype=np.float32)

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
    color_std=np.zeros(shape=(len(y),len(x)))
    color_bavg=np.zeros(shape=(len(y),len(x)))
    color_gavg=np.zeros(shape=(len(y),len(x)))
    color_ravg=np.zeros(shape=(len(y),len(x)))

    #color_flag=np.zeros(shape=(len(y),len(x)))
    for i in range(len(y)):
        for j in range(len(x)):
            color_bstd[i][j]=np.std(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_gstd[i][j]=np.std(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_rstd[i][j]=np.std(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_std[i][j]=color_bstd[i][j]+color_gstd[i][j]+color_rstd[i][j]
            color_bavg[i][j]=np.mean(b[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_gavg[i][j]=np.mean(g[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            color_ravg[i][j]=np.mean(r[y[i]-step/2:y[i]+step/2,x[j]-step/2:x[j]+step/2])
            #if color_b[i][j]+color_g[i][j]+color_r[i][j]>100:
                #color_flag[i][j]=1
                #img = cv2.rectangle(img,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)
    return color_std,color_bavg,color_gavg,color_ravg

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

'''online ground homography calculation'''
'''get better homography value!!!'''
'''to get the precise focal length later'''
def findGH(img1,img2,LRBB):

    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    img1[0:270,0:640]=np.zeros((270,640),dtype=np.uint8)
    img2[0:270,0:640]=np.zeros((270,640),dtype=np.uint8)

    MIN_MATCH_COUNT = 10

    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        print(H)

    else:
        if LRBB==0:
            H=H_L
        elif LRBB==1:
            H=H_R
        elif LRBB==2:
            H=H_B
        else:
            H=H_BB
        print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
        matchesMask = None

    return H

def Homography(h,w,prev,imgL,imgR,imgB,imgBB,Ground_truth):
    #change to robot's camera motion later
    bp,gp,rp = cv2.split(prev)
    bl,gl,rl = cv2.split(imgL)
    br,gr,rr = cv2.split(imgR)
    bb,gb,rb = cv2.split(imgB)
    bbb,gbb,rbb = cv2.split(imgBB)

    #print(H_L)
    #print(H_R)
    for i in range(w):
        for j in range(h):
            spt=np.float32([[i,j]]).reshape(-1,1,2)
            dptL=cv2.perspectiveTransform(spt,H_L)
            dptL=np.int32(dptL)
            dptR=cv2.perspectiveTransform(spt,H_R)
            dptR=np.int32(dptR)
            dptB=cv2.perspectiveTransform(spt,H_B)
            dptB=np.int32(dptB)
            dptBB=cv2.perspectiveTransform(spt,H_BB)
            dptBB=np.int32(dptBB)

            '''better filters needed, std,brightness,etc.'''
            if dptL[0][0][0] in range(w) and dptL[0][0][1] in range(h):
                if (abs(bl[dptL[0][0][1]][dptL[0][0][0]]-bp[j][i])<=15 and
                    abs(gl[dptL[0][0][1]][dptL[0][0][0]]-gp[j][i])<=15 and
                    abs(rl[dptL[0][0][1]][dptL[0][0][0]]-rp[j][i])<=15):
                    prev[j][i]=[0,255,0]
                    imgL[j][i]=[0,255,0]
                    Ground_truth[j][i]=1
            if dptR[0][0][0] in range(w) and dptR[0][0][1] in range(h):
                if (abs(br[dptR[0][0][1]][dptR[0][0][0]]-bp[j][i])<=15 and
                    abs(gr[dptR[0][0][1]][dptR[0][0][0]]-gp[j][i])<=15 and
                    abs(rr[dptR[0][0][1]][dptR[0][0][0]]-rp[j][i])<=15):
                    prev[j][i]=[0,255,0]
                    imgR[j][i]=[0,255,0]
                    Ground_truth[j][i]=1
            if dptB[0][0][0] in range(w) and dptB[0][0][1] in range(h):
                if (abs(bb[dptB[0][0][1]][dptB[0][0][0]]-bp[j][i])<=15 and
                    abs(gb[dptB[0][0][1]][dptB[0][0][0]]-gp[j][i])<=15 and
                    abs(rb[dptB[0][0][1]][dptB[0][0][0]]-rp[j][i])<=15):
                    prev[j][i]=[0,255,0]
                    imgB[j][i]=[0,255,0]
                    Ground_truth[j][i]=1
            if dptBB[0][0][0] in range(w) and dptBB[0][0][1] in range(h):
                if (abs(bbb[dptBB[0][0][1]][dptBB[0][0][0]]-bp[j][i])<=15 and
                    abs(gbb[dptBB[0][0][1]][dptBB[0][0][0]]-bp[j][i])<=15 and
                    abs(rbb[dptBB[0][0][1]][dptBB[0][0][0]]-rp[j][i])<=15):
                    prev[j][i]=[0,255,0]
                    imgBB[j][i]=[0,255,0]
                    Ground_truth[j][i]=1

    for n in range(6):
        for i in range(1,w-1):
            for j in range(1,h-1):
                if (Ground_truth[j][i]==0 and
                   np.sum(Ground_truth[j-1:j+2,i-1:i+2])>=4):
                   Ground_truth[j][i]==1
                   prev[j][i]=[0,255,0]


    return prev,imgL,imgR,imgB,imgBB,Ground_truth


i=0
while i<1:

    #initiate the two images
    cam = cv2.VideoCapture(0)
    n=0
    while n<10:
        ret, prev = cam.read()
        n=n+1

    capR=input('press 1 if the camera is oriented right properly')
    if capR==1:
        while 1:
            ret, imgR = cam.read()
            cv2.imshow('imgR',imgR)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    capL=input('press 1 if the camera is oriented left properly')
    if capL==1:
        while 1:
            ret, imgL = cam.read()
            cv2.imshow('imgL',imgL)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    capB=input('press 1 if the camera is moved back properly')
    if capB==1:
        while 1:
            ret, imgB = cam.read()
            cv2.imshow('imgB',imgB)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    capBB=input('press 1 if the camera is moved further back properly')
    if capBB==1:
        while 1:
            ret, imgBB = cam.read()
            cv2.imshow('imgBB',imgBB)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


    step=16;
    h,w=prev.shape[:2]
    Ground_truth=np.zeros(shape=(h,w))
    y=np.linspace(step/2,h-step/2,int((h-step)/step),dtype=int)
    x=np.linspace(step/2,w-step/2,int((w-step)/step),dtype=int)

    bri_std,bri_mean=bri(prev,step,h,w,y,x)
    color_std,color_bavg,color_gavg,color_ravg=color(prev,step,h,w,y,x)

    H_L = findGH(prev,imgL,0)
    H_R = findGH(prev,imgR,1)
    H_B = findGH(prev,imgB,2)
    H_BB = findGH(prev,imgBB,3)
    prev,imgL,imgR,imgB,imgBB,Ground_truth=Homography(h,w,prev,imgL,imgR,imgB,imgBB,Ground_truth)

    '''need to group pixels based on continuaty'''
    '''need to deal with the featureless regions that are not ground'''

    #pixels in the lower middle part of the image represent the ground
    #print(bri)
    g_cri_bstd=np.sum(bri_std[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_bavg=np.sum(bri_mean[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cstd=np.sum(color_std[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cbavg=np.sum(color_bavg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cgavg=np.sum(color_gavg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9
    g_cri_cravg=np.sum(color_ravg[len(y)-4:len(y)-1,len(x)/2-1:len(x)/2+2])/9


    #g_cri_homo=sum(Homo[len(y)-5:len(y)-1,len(x)/2-3:len(x)/2+1])/16
    #print(Homo[len(y)-5:len(y)-1,len(x)/2-3:len(x)/2+1])
    '''g_cri_flow=(flow[len(y)-1][len(x)/2-1][0]+flow[len(y)-1][len(x)/2-2][0]+flow[len(y)-1][len(x)/2][0]+
                flow[len(y)-2][len(x)/2-1][0]+flow[len(y)-2][len(x)/2-2][0]+flow[len(y)-2][len(x)/2][0])/6
    print(g_cri_flow)'''

    #compare, decide and draw the ground regions
    '''for i in range(len(y)):
        for j in range(len(x)):
            #weighted method...optical flow has less impact for the pixels far away,can be used just for simple test
            if (abs(bri_std[i][j]-g_cri_bstd)<50 and abs(bri_mean[i][j]-g_cri_bavg)<50
                and abs(color_std[i][j]-g_cri_cstd)<50 and abs(color_bavg[i][j]-g_cri_cbavg)<50
                and abs(color_gavg[i][j]-g_cri_cgavg)<50 and abs(color_ravg[i][j]-g_cri_cravg)<50):
                #and abs(flow[i][j][0]-g_cri_flow)<0.8:
            #if color_flag[i][j]==1 and bri_flag[i][j]==1:
                prev = cv2.rectangle(prev,(x[j]-step/2,y[i]-step/2),(x[j]+step/2,y[i]+step/2),(0,0,255),3)'''
    cv2.imshow('prev',prev)
    cv2.imshow('imgL',imgL)
    cv2.imshow('imgR',imgR)
    cv2.imshow('imgB',imgB)
    cv2.imshow('imgBB',imgBB)
    cv2.imwrite('prev.jpg',prev)
    cv2.imwrite('imgL.jpg',imgL)
    cv2.imwrite('imgR.jpg',imgR)
    cv2.imwrite('imgB.jpg',imgB)
    cv2.imwrite('imgBB.jpg',imgBB)

    i=i+1
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
