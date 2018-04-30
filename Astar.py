# Enter your code here. Read input from STDIN. Print output to STDOUT

import numpy as np

class Node:
    def __init__(self,value,point):
        self.value = value
        self.point = point
        self.parent = None
        self.H = 0
        self.G = 0

    def move_cost(self,other):
        return 0 if self.value == '.' else 1

def children(point,grid):
    x,y = point.point
    if x==0 and y==0:
        links = [grid[d[0]][d[1]] for d in [(x + 1, y),(x,y + 1)]]
    elif x==0 and y==len(grid[x]):
        links = [grid[d[0]][d[1]] for d in [(x + 1, y),(x,y - 1)]]
    elif x==len(grid) and y==0:
        links = [grid[d[0]][d[1]] for d in [(x - 1, y),(x,y + 1)]]
    elif x==len(grid) and y==len(grid[x]):
        links = [grid[d[0]][d[1]] for d in [(x - 1, y),(x,y - 1)]]
    elif x==0:
        links = [grid[d[0]][d[1]] for d in [(x + 1, y),(x,y - 1),(x,y + 1)]]
    elif x==len(grid):
        links = [grid[d[0]][d[1]] for d in [(x - 1, y),(x,y - 1),(x,y + 1)]]
    elif y==0:
        links = [grid[d[0]][d[1]] for d in [(x,y + 1),(x - 1,y),(x + 1,y)]]
    elif y==len(grid[x]):
        links = [grid[d[0]][d[1]] for d in [(x,y - 1),(x - 1,y),(x + 1,y)]]
    else:
        links = [grid[d[0]][d[1]] for d in [(x - 1,y),(x,y - 1),(x,y + 1),(x + 1,y)]]
    n=0
    while 1:
        ind=len(links)
        if n>=ind:
            break
        if links[n].value=='%':
            del links[n]
            n-=1
        n+=1
    return links
def manhattan(point,point2):
    return np.sqrt(abs(point.point[0] - point2.point[0])**2 + abs(point.point[1]-point2.point[1])**2)
def aStar(start, goal, grid):
    #The open and closed sets
    openset = set()
    closedset = set()
    #Current point is the starting point
    current = start
    #Add the starting point to the open set
    openset.add(current)
    #While the open set is not empty
    while openset:
        #Find the item in the open set with the lowest G + H score
        current = min(openset, key=lambda o:o.G + o.H)
        #If it is the item we want, retrace the path and return it
        if current == goal:
            path = []
            while current.parent:
                path.append(current)
                current = current.parent
            path.append(current)
            return path[::-1]
        #Remove the item from the open set
        openset.remove(current)
        #Add it to the closed set
        closedset.add(current)
        #Loop through the node's children/siblings
        for node in children(current,grid):
            #If it is already in the closed set, skip it
            if node in closedset:
                continue
            #Otherwise if it is already in the open set
            if node in openset:
                #Check if we beat the G score
                new_g = current.G + current.move_cost(node)
                if node.G > new_g:
                    #If so, update the node to have a new parent
                    node.G = new_g
                    node.parent = current
            else:
                #If it isn't in the open set, calculate the G and H score for the node
                node.G = current.G + current.move_cost(node)
                node.H = manhattan(node, goal)
                #Set the parent to our current item
                node.parent = current
                #Add it to the set
                openset.add(node)
    #Throw an exception if there is no path
    raise ValueError('No Path Found')

def next_move(pacman,food,grid):
    #Convert all the points to instances of Node
    grid_node=[[None for _ in range(len(grid))] for _ in range(len(grid[0]))]
    for x in xrange(len(grid)):
        for y in xrange(len(grid[x])):
            grid_node[x][y] = Node(grid[x][y],(x,y))
    #Get the path
    path = aStar(grid_node[pacman[0]][pacman[1]],grid_node[food[0]][food[1]],grid_node)
    #Output the path
    plength=len(path) - 1
    print(plength)
    flag=0
    for node in path:
        x, y = node.point
        if flag==0:
            px,py=x,y
            flag=1
        else:
            px,py=np.append(px,x),np.append(py,y)
        #print(x, y)

    return px,py,plength

if __name__ == '__main__':

    grid=[[None for _ in range(10)] for _ in range(10)]
    next_move([0,0],[5,5],grid)
