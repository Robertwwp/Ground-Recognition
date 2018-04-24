'''
A* search based on neighbors
probalistic map
map def based on image
'''
'''ltl2ba on the path, tulip interface used'''

from tulip.interfaces import ltl2ba
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from Astar import next_move
import networkx as nx
import numpy as np


#Define the map and get grid map for A*
plt.xlim(0,100),plt.ylim(0,100)
#plt.scatter([1,1],[1,2])
axis=plt.gca()

box_size=4
StartM,GoalM1,GoalM2,obsM1=[(0,0),box_size,box_size],[(68,68),box_size,box_size],\
                           [(78,38),box_size,box_size],[(20,20),20,40]
for p in [
        patches.Rectangle(StartM[0],StartM[1],StartM[2],facecolor="red"),
        patches.Rectangle(GoalM1[0],GoalM1[1],GoalM1[2],facecolor="green"),
        patches.Rectangle(GoalM2[0],GoalM2[1],GoalM2[2],facecolor="green"),
        patches.Rectangle(obsM1[0],obsM1[1],obsM1[2],facecolor="black")
    ]:

    axis.add_patch(p)

grid=[[None for _ in range(100)] for _ in range(100)]
for i in range(obsM1[1]):
    for j in range(obsM1[2]):
        grid[obsM1[0][0]+i][obsM1[0][1]+j]='%'
for i in range(box_size):
    for j in range(box_size):
        grid[StartM[0][0]+i][StartM[0][1]+j]='start'
        grid[GoalM1[0][0]+i][GoalM1[0][1]+j]='Goal1'
        grid[GoalM2[0][0]+i][GoalM2[0][1]+j]='Goal2'

#check observations
def check_obsv(x,y):

    obs,g1,g2=0,0,0
    if obsM1[0][0]<=x<=obsM1[0][0]+obsM1[1] and obsM1[0][1]<=y<=obsM1[0][1]+obsM1[2]:
        obs=1
    if GoalM1[0][0]<=x<=GoalM1[0][0]+GoalM1[1] and GoalM1[0][1]<=y<=GoalM1[0][1]+GoalM1[2]:
        g1=1
    if GoalM2[0][0]<=x<=GoalM2[0][0]+GoalM2[1] and GoalM2[0][1]<=y<=GoalM2[0][1]+GoalM2[2]:
        g2=1

    return obs,g1,g2

#def check_goal():


#ltl constraints
parser = ltl2ba.Parser()
f = '[] !obs && []<> g1 && []<> g2'
out = ltl2ba.call_ltl2ba(f)
print(out)
symbols, g, initial, accepting = parser.parse(out)
states=g.nodes()
print(g.neighbors(states[0]))
#print(g[states[1]][states[2]][0]['guard'])
obs,g1,g2,g3=0,1,0,0
trans=g[states[1]][states[2]][0]['guard']
print(trans)
if eval(trans):print('true')
else: print('false')


'''get observations and check state change'''
'''modify A* heuristic'''

#A* searching
#################################################
#velocity m/s, sampling time, scale on map
'''v,t,scale=0.5,0.2,10

#start,Goal=[0,0],[70,70]
#px,py=next_move(start,Goal,grid)
rx,ry,pose,m=2,2,0,0  #pose is the head dir wrt x axis
T,cur_state=200,states[0]   #simulation ticks, time=T*t
for m in range(T):
    obs,g1,g2=check_obsv(rx,ry)
    #update state
    for neighbor in g.neighbors(cur_state) if neighbor!=cur_state:
        trans=g[cur_state][neighbor][0]['guard']
        if eval(trans):
            cur_state=neighbor


    if 68<=rx<=72 and 68<=ry<=72:  #should be slightly larger than the actual area
        break
    for n in range(45):
        theta=((n*2+(-45))/180.0)*np.pi
        rx_,ry_=rx+scale*t*v*np.cos(pose+theta),ry+scale*t*v*np.sin(pose+theta)
        cost_=np.min(abs(rx_-px)+abs(ry_-py))+abs(rx_-Goal[0])+abs(ry_-Goal[1])
        if n==0: cost=cost_
        else: cost=np.append(cost,cost_)

    theta=((np.where(cost==np.min(cost))[0][0]*2+(-45))/180.0)*np.pi
    rx,ry=rx+scale*t*v*np.cos(pose+theta),ry+scale*t*v*np.sin(pose+theta)
    pose+=theta
    if m==0: rx_full,ry_full=rx_,ry_
    else: rx_full,ry_full=np.append(rx_full,rx),np.append(ry_full,ry)

plt.plot(px,py)
plt.plot(rx_full,ry_full,'orange')

plt.show()'''