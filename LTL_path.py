'''LTL Constrained A* Planning for Mapping an Indoor Environment'''

from tulip.interfaces import ltl2ba
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon,Point,LineString
from descartes import PolygonPatch
from Astar import next_move
import networkx as nx
import numpy as np
import random

#define and draw the map
#further work needed to transfer the ground recognition result to the map here
plt.xlim(0,100),plt.ylim(0,100)
#plt.scatter([1,1],[1,2])
axis=plt.gca()

box_size,safe_rng=6,3 #size of the start and goal region; the safe range for obstacles
poly=Polygon([(0,35),(0,100),(100,100),(100,30),(80,60),(40,65),(0,25)]) #non-ground area

StartM,GoalM1,GoalM2,GoalM3,obsM1=[(47,0),box_size,box_size],[(10,20),box_size,box_size],\
                                  [(47,47),box_size,box_size],[(84,20),box_size,box_size],[(30,15),10,10]
GMset,Gset_full,obs_full=[StartM,GoalM1,GoalM2,GoalM3],['s','g1','g2','g3'],['s','g1','g2','g3','obs']
for p in [
        patches.Rectangle(StartM[0],StartM[1],StartM[2],facecolor="red"),
        patches.Rectangle(GoalM1[0],GoalM1[1],GoalM1[2],facecolor="green"),
        patches.Rectangle(GoalM2[0],GoalM2[1],GoalM2[2],facecolor="green"),
        patches.Rectangle(GoalM3[0],GoalM3[1],GoalM3[2],facecolor="green"),
        patches.Rectangle((obsM1[0][0]+safe_rng,obsM1[0][1]+safe_rng),obsM1[1]-2*safe_rng,obsM1[2]-2*safe_rng,facecolor="black"),
        patches.Rectangle(obsM1[0],obsM1[1],obsM1[2],facecolor="black",alpha=0.1),
        PolygonPatch(poly)
    ]:

    axis.add_patch(p)

#get the grid map for A*, label all the un-passible areas
grid=[[None for _ in range(100)] for _ in range(100)]
for i in range(obsM1[1]):
    for j in range(obsM1[2]):
        grid[obsM1[0][0]+i][obsM1[0][1]+j]='%'
for i in range(100):
    for j in range(100):
        if poly.contains(Point(i,j)):
            grid[i][j]='%'


############################################################################
#ltl constraints
parser = ltl2ba.Parser()
f = '[]<> s && <> g1 && <> g2 && <> g3 && [] !obs'
#f = '[]<> s && <> (g3 && <> (g1 && <> g2)) && [] !obs '
out = ltl2ba.call_ltl2ba(f)
print(out)
symbols, g, initial, accepting = parser.parse(out)
states=g.nodes()

#get observations and check state change
#modify A* heuristic

#check observations
def check_obsv(x,y):

    obs,g1,g2,g3,s=0,0,0,0,0
    if obsM1[0][0]<=x<=obsM1[0][0]+obsM1[1] and obsM1[0][1]<=y<=obsM1[0][1]+obsM1[2]:
        obs=1
    if GoalM1[0][0]<=x<=GoalM1[0][0]+GoalM1[1] and GoalM1[0][1]<=y<=GoalM1[0][1]+GoalM1[2]:
        g1=1
    if GoalM2[0][0]<=x<=GoalM2[0][0]+GoalM2[1] and GoalM2[0][1]<=y<=GoalM2[0][1]+GoalM2[2]:
        g2=1
    if GoalM3[0][0]<=x<=GoalM3[0][0]+GoalM3[1] and GoalM3[0][1]<=y<=GoalM3[0][1]+GoalM3[2]:
        g3=1
    if StartM[0][0]<=x<=StartM[0][0]+StartM[1] and StartM[0][1]<=y<=StartM[0][1]+StartM[2]:
        s=1

    return obs,g1,g2,g3,s

def check_goals(state):

    Gset,obs=[],0
    for neighbor in g.neighbors(state):
        if neighbor!=cur_state:
            trans=g[state][neighbor][0]['guard']
            for n in range(len(Gset_full)):
                exec("%s = %d" % (Gset_full[n],1))
                if eval(trans):
                    Gset.insert(len(Gset),GMset[n])
                exec("%s = %d" % (Gset_full[n],0))

    return Gset

#actual path finding
#################################################
#velocity m/s, sampling period, scale factor on map
v,t,scale=0.5,0.3,10

#rx, ry indicates teh start position
#pose is the head direction wrt x axis
rx,ry,pose,m,flag,flag_g=50,3,0,0,0,0
T,cur_state=200,states[0]   #simulation ticks, time=T*t; initialize the current state
#the initial state might not be state[0] for some automaton, try find the words 'init' in the future

for m in range(T):

    #check observations every iteration
    obs,g1,g2,g3,s=check_obsv(rx,ry)
    #first step
    if flag==0:
        Gset=check_goals(cur_state)
        distance,Goal=np.inf,None
        for i in range(len(Gset)):
            xG,yG=Gset[i][0][0],Gset[i][0][1]
            if np.sqrt(abs(xG-rx)**2+abs(yG-ry)**2)<distance:
                distance=np.sqrt(abs(xG-rx)**2+abs(yG-ry)**2)
                Goal=Gset[i][0]
                Goal=(Goal[0]+box_size/2,Goal[1]+box_size/2)
        if Goal==None:
            print('no more goals')
            flag_g=1
        else:
            px,py,plength=next_move((int(rx),int(ry)),Goal,grid)
            if plength<=5: flag_g=1 #no self loop goal
            #plt.plot(px,py)
            flag=1
    else:
        for neighbor in g.neighbors(cur_state):
            if neighbor!=cur_state:
                trans=g[cur_state][neighbor][0]['guard']
                #check transition, if state change, find new goal and new path
                if eval(trans):
                    cur_state=neighbor
                    Gset=check_goals(cur_state)
                    #find nearest goal
                    distance,Goal=np.inf,None
                    for i in range(len(Gset)):
                        xG,yG=Gset[i][0][0],Gset[i][0][1]
                        if np.sqrt(abs(xG-rx)**2+abs(yG-ry)**2)<distance:
                            distance=np.sqrt(abs(xG-rx)**2+abs(yG-ry)**2)
                            Goal=Gset[i][0]
                            Goal=(Goal[0]+box_size/2,Goal[1]+box_size/2)
                    if Goal==None:
                        print('no more goals')
                        flag_g=1
                        break
                    else:
                        print('state change')
                        print(cur_state)
                        px,py,plength=next_move((int(rx),int(ry)),Goal,grid)
                        if plength<=5: flag_g=1 #no self loop goal
                        #plt.plot(px,py)
                        break


    if flag_g==0:
    #get trajectory based on kinematics
        random.seed()
        uncertainty=(random.uniform(0,2),random.uniform(0,2)) #uncertainty on x and y direction
        for n in range(45):
            theta=((n*2+(-45))/180.0)*np.pi
            rx_,ry_=rx+scale*t*v*np.cos(pose+theta)+uncertainty[0],ry+scale*t*v*np.sin(pose+theta)+uncertainty[1]
            cost_=np.min(np.sqrt(abs(rx_-px)**2+abs(ry_-py)**2))+np.sqrt(abs(rx_-Goal[0])**2+abs(ry_-Goal[1])**2)
            if n==0: cost=cost_
            else: cost=np.append(cost,cost_)

        theta=((np.where(cost==np.min(cost))[0][0]*2+(-45))/180.0)*np.pi
        rx,ry=rx+scale*t*v*np.cos(pose+theta),ry+scale*t*v*np.sin(pose+theta)
        pose+=theta
        if m==0: rx_full,ry_full=rx_,ry_
        else: rx_full,ry_full=np.append(rx_full,rx),np.append(ry_full,ry)
    else:
        break

plt.plot(rx_full,ry_full,'orange')

plt.show()
