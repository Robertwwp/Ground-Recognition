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

for p in [
        patches.Rectangle((0, 0),4,4,facecolor="red"),
        patches.Rectangle((68, 68),4,4,facecolor="green"),
        patches.Rectangle((78, 38),4,4,facecolor="green"),
        patches.Rectangle((20, 20),20,40,facecolor="black")
    ]:

    axis.add_patch(p)

grid=[[None for _ in range(100)] for _ in range(100)]
for i in range(20):
    for j in range(40):
        grid[20+i][20+j]='%'
for i in range(4):
    for j in range(4):
        grid[0+i][0+j]='start'
        grid[68+i][68+j]='Goal1'
        grid[78+i][38+j]='Goal2'

Gset=[[70,70],[80,40]]

#ltl constraints
parser = ltl2ba.Parser()
f = '[] !obs && []<> g1 && []<> g2'
out = ltl2ba.call_ltl2ba(f)
print(out)
symbols, g, initial, accepting = parser.parse(out)
states=g.nodes()
print(g.get_edge_data(states[1],states[2]))
print(g.neighbors(states[2]))

'''get observations and check state change'''
'''modify A* heuristic'''

#A* searching
#################################################
#velocity m/s, sampling time, scale on map
v,t,scale=1,0.1,10

start,Goal=[0,0],[70,70]
px,py=next_move(start,Goal,grid)
rx,ry,pose,m=0,0,0,0  #pose is the head dir wrt x axis
for m in range(len(px)):
    if 68<=rx<=72 and 68<=ry<=72:
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

plt.show()
