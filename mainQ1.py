from queue import PriorityQueue
import numpy as np
import pandas as pd
import networkx as nx
import sys
import time
import math
from collections import defaultdict

def getIndex(name,arry):
    for i in range(len(arry)):
        if name==arry[i]:
            return i
    return -1

def getRow(path,type,matrix,dataFrame):
    rowArry=[]
    if type==1:
        for index in range(len(path) - 1):
            rowIndex = matrix["index"][path[index]][path[index + 1]]
            rowArry .append(dataFrame.iloc[rowIndex])
    else:
        for index in range(len(path) - 1):
            row=df.loc[(df['SourceAirport'] == path[index]) & (df['DestinationAirport'] ==path[index + 1])].iloc[0]
            rowArry.append(row)
    return rowArry

def printFile(path,dataFrame,matrix,type,startTime):
    exTime=time.time() - startTime
    if(type==1):
        file = open("11-UIAI4021-PR1-Q1.txt", "w+")
        file.write("dijkstra Algorithm\n")
        file.write("Eecution Time: %f s\n-----------------------------\n" % (exTime))
    else:
        file = open("11-UIAI4021-PR2-Q1.txt", "w+")
        file.write("A* Algorithm\n")
        file.write("Eecution Time: %f s\n-----------------------------\n" % (exTime))
    rowArry = getRow(path, type, matrix, dataFrame)

    numFlight,tDuration,tPrice,tTime=0,0,0,0
    for index in range(len(path)-1):
        numFlight+=1
        row=rowArry[index]
        file.write("Flight #%d\n"%(numFlight))
        file.write("from :%s _ %s , %s\n"%(row["SourceAirport_City"],row["SourceAirport"],row["SourceAirport_Country"]))
        file.writelines("to :%s _ %s , %s\n"%(row["DestinationAirport_City"],row["DestinationAirport"], row["DestinationAirport_Country"]))
        file.writelines("Duration: %d km\n"%(math.ceil(row["Distance"])))
        tDuration+=row["Distance"]
        file.writelines("Time: %d h\n"%(math.ceil(row["FlyTime"])))
        tTime+=row["FlyTime"]
        file.writelines("Price: %d $\n"%(math.ceil(row["Price"])))
        tPrice+=row["Price"]
        file.writelines("-----------------------------\n")
    file.writelines("Total Price: %d $\nTotal Duration: %d km\nTotal Time: %d h"%(math.ceil(tPrice),math.ceil(tDuration),math.ceil(tTime)))
    file.close()
class Dijkstra:
    def dijkstra(self, matrix, startVertex, destinationVertex):
        numVertices = len(matrix[0])
        shortestDistances = [sys.maxsize] * numVertices
        touch = [False] * numVertices
        for vertexIndex in range(numVertices):
            shortestDistances[vertexIndex] = sys.maxsize
            touch[vertexIndex] = False
        shortestDistances[startVertex] = 0
        parents = [-1] * numVertices
        parents[startVertex] = -1
        for i in range(1, numVertices):
            nearestVertex = -1
            shortest_distance = sys.maxsize
            for vertexIndex in range(numVertices):
                if not touch[vertexIndex] and shortestDistances[vertexIndex] < shortest_distance:
                    nearestVertex = vertexIndex
                    shortest_distance = shortestDistances[vertexIndex]
            touch[nearestVertex] = True
            for vertexIndex in range(numVertices):
                edge_distance = matrix[nearestVertex][vertexIndex]
                if edge_distance > 0 and shortest_distance + edge_distance < shortestDistances[vertexIndex]:
                    parents[vertexIndex] = nearestVertex
                    shortestDistances[vertexIndex] = shortest_distance + edge_distance
        path = []
        self.print_path(destinationVertex, parents, path)
        return path

    def print_path(self, current_vertex, parents, path):
        if current_vertex == -1:
            return
        self.print_path(parents[current_vertex], parents, path)
        path.append(current_vertex)

class Graph():
    def __init__(self):
        self.edges = defaultdict(list)

    def addEdge(self, fromVerex, toVertex, weight,index):
        if toVertex in self.edges[fromVerex]:
            return
        self.edges[fromVerex].append((toVertex,weight,index))
    def dfToGraph(self):
        for i in range(df.shape[0]):
            row=df.iloc[i]
            self.addEdge(row["SourceAirport"],row["DestinationAirport"],row["Distance"],i)
class aStar:
    def __init__(self, adjac_lis):
        self.adjac_lis = adjac_lis
    def get_neighbors(self, v):
        try:
         return self.adjac_lis[v]
        except KeyError:
            return []

    # This is heuristic function which is having equal values for all nodes
    def heuristic(self,n, stopH, stopL, stopW):
        neighbor = self.get_neighbors(n)
        if len(neighbor) == 0:
            return math.inf
        (m, weight, index) = neighbor[0]
        rowN = df.iloc[index]
        nH=rowN["SourceAirport_Altitude"]
        nL=rowN["SourceAirport_Longitude"]
        nW=rowN["SourceAirport_Latitude"]
        h=(stopH-nH)**2
        l=(stopL-nL)**2
        w=(stopW-nW)**2
        heuristic=math.sqrt(h+l+w)
        return heuristic

    def a_star_algorithm(self, start, stop):
        open_lst = set([start])
        closed_lst = set([])
        g = {}
        g[start] = 0
        par = {}
        par[start] = start
        rowS=df.loc[(df['DestinationAirport']==stop)].iloc[0]
        stopH=rowS["DestinationAirport_Altitude"]
        stopL=rowS["DestinationAirport_Longitude"]
        stopW=rowS["DestinationAirport_Latitude"]
        queue = PriorityQueue()
        queue.put((self.heuristic(start,stopH,stopL,stopW), start))
        while queue:
            (f,n)=queue.get()
            if n == None:
                print('Path does not exist!')
                return None
            if n == stop:
                reconst_path = []
                while par[n] != n:
                    reconst_path.append(n)
                    n = par[n]
                reconst_path.append(start)
                reconst_path.reverse()
                return reconst_path
            for (m ,weight,index) in self.get_neighbors(n):
                if m not in open_lst and m not in closed_lst:
                    open_lst.add(m)
                    par[m] = n
                    g[m] = g[n] + weight
                    heuristicM=self.heuristic(m,stopH,stopL,stopW)
                    queue.put((heuristicM+g[m],m))
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        par[m] = n
                        if m in closed_lst:
                            closed_lst.remove(m)
                            heuristic = self.heuristic(m,stopH,stopL,stopW)
                            open_lst.add(m)
                            queue.put((heuristic+g[m], m))
            closed_lst.add(n)
            open_lst.remove(n)

        print('Path does not exist!')
        return None

df = pd.read_csv('./Flight_Data.csv')
inp=input()
inp=inp.split(" - ")
startDijkstra = time.time()
df["index"]=df.index
Graphtype = nx.DiGraph()
G = nx.from_pandas_edgelist(df,source="SourceAirport",target="DestinationAirport",edge_attr=["Distance","index"], create_using=Graphtype)
indexOrgin=getIndex(inp[0],list(Graphtype.nodes))
indexDestination=getIndex(inp[1],list(Graphtype.nodes))
dtype = np.dtype([("Distance",int),("index",int)])
mtrx=nx.to_numpy_array(G, nonedge=0,dtype=dtype,weight=None)
dijkstra=Dijkstra()
path=dijkstra.dijkstra(mtrx["Distance"],indexOrgin,indexDestination)
printFile(path,df,mtrx,1,startDijkstra)
graphA=Graph()
graphA.dfToGraph()
startA = time.time()
aStar = aStar(dict(graphA.edges))
path=aStar.a_star_algorithm(inp[0],inp[1])
printFile(path,df,[],2,startA)