#!/usr/bin/python3

from CS312Graph import *
from abc import ABC, abstractmethod

import time
import math

class PriorityQueue(ABC):
    """
    Abstract base class for priority queue implementations. 
    """

    def __init__(self):
        """Initializes the priority queue"""
        self.queue = None

    @abstractmethod
    def makeQueue(self, nodes, startNode):
        """Creates a priority queue from a list of nodes"""
        pass

    @abstractmethod
    def insert(self, node, dist):
        """Inserts a node into the priority queue with a given distance value"""
        pass
    
    @abstractmethod
    def deleteMin(self, dist):
        """Deletes the node with the minimum distance value from the priority queue"""
        pass
    
    @abstractmethod
    def decreaseKey(self, node, dist):
        """Decreases the distance value of a node and adjusts the priority queue"""
        pass

class PQArray(PriorityQueue):
    def __init__(self, nodes):
        self.queue = []
        self.makeQueue(nodes)

    # Time: O(n)
    def makeQueue(self, nodes):
        for node in nodes:
            self.insert(node)  

    # Time: O(1)        
    def insert(self, node):
        self.queue.append(node.node_id)  

    # Time: O(n) 
    def deleteMin(self, dist):
        if not self.queue:
            return None  # Handle the case when queue is empty
        
        minIndex = self.queue[0]

        for node in self.queue:
            if dist[node] < dist[minIndex]:
                minIndex = node

        self.queue.remove(minIndex)
        return minIndex
 
    # Time: O(1)
    def decreaseKey(self, node, dist):
        pass

class PQHeap(PriorityQueue):
    def __init__(self, nodes, startNode):
        self.heap = []
        self.nodePosition = []
        self.makeQueue(nodes, startNode)

    def makeQueue(self, nodes, startNode):   
        self.heap.append(startNode)
        index = 1
        for node in nodes:
            nodeId = node.node_id
            if nodeId != startNode:
                self.heap.append(nodeId)
                self.nodePosition.append(index)
            else:
                self.nodePosition.append(0)

            index += 1


    def insert(self, node, dist):
        self.heap.append(node.node_id)
        self.bubbleUp(node.node_id, (len(self.heap) - 1), dist)
    
    def bubbleUp(self, x, i, dist):
        p = i // 2
        while i > 0 and dist[self.heap[p]] > dist[x]:
            self.heap[i] = self.heap[p]
            self.heap[p] = x

            temp = self.nodePosition[self.heap[i]]
            self.nodePosition[self.heap[i]] = self.nodePosition[self.heap[p]]
            self.nodePosition[self.heap[p]] = temp

            i = p
            p = i // 2

    def decreaseKey(self, node, dist):
        i = self.nodePosition[node]
        if i < len(self.heap):
            self.bubbleUp(node, i, dist)  # O(log(n))


    def deleteMin(self, dist):  # O(log(n))
        if len(self.heap) == 0:
            return None
        else:
            minDist = self.heap[0]
            if len(self.heap) > 1:
                self.heap[0] = self.heap[-1]
                self.nodePosition[self.heap[-1]] = 0
                self.heap.pop()
                self.nodePosition[minDist] = len(self.heap)
                self.siftdown(self.heap[0], 0, dist)

            return minDist

    def siftdown(self, x, i, dist):  # O(log(n))
        minIndex = self.minchild(i, dist)  # O(1)
        while minIndex != 0 and dist[self.heap[minIndex]] < dist[x]:
            self.heap[i] = self.heap[minIndex]
            self.heap[minIndex] = x

            temp = self.nodePosition[self.heap[i]]
            self.nodePosition[self.heap[i]] = self.nodePosition[self.heap[minIndex]]
            self.nodePosition[self.heap[minIndex]] = temp

            i = minIndex
            minIndex = self.minchild(i, dist)

    def minchild(self, i, dist):
        firstChildIdx = (2*i)+1
        secondChildIdx = (2*i)+2
        if secondChildIdx < len(self.heap) and firstChildIdx < len(self.heap):
            if dist[self.heap[firstChildIdx]] < dist[self.heap[secondChildIdx]]:
                return firstChildIdx
            else:
                return secondChildIdx
        elif firstChildIdx < len(self.heap):
            return firstChildIdx
        else:
            return -1
        
class NetworkRoutingSolver:
    def __init__(self):
        pass

    def initializeNetwork(self, network):
        assert(type(network) == CS312Graph)
        self.network = network

    def getShortestPath(self, destIndex):
        path = []
        length = 0
        i = destIndex

        while i != self.source:
            j = self.prev[i]
            if j is not None:
                edge = next(e for e in self.network.nodes[j].neighbors if e.dest.node_id == i)
      
                if edge:
                    path.insert(0, (edge.src.loc, edge.dest.loc, '{:.0f}'.format(edge.length)))
                    length += edge.length
                else:
                    print("UNREACHABLE")
                    length = math.inf
                    break
                i = j

        return {'cost': length, 'path': path}

    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time() 
        self.dijkstra(self.source, use_heap)
        t2 = time.time()
        return t2 - t1 

    def dijkstra(self, startNode, use_heap):
        numNodes = len(self.network.nodes)
        
        self.dist = [math.inf] * numNodes
        self.prev = [None] * numNodes
        self.dist[startNode] = 0

        pq = PQHeap(self.network.nodes, startNode) if use_heap else PQArray(self.network.nodes)

        while numNodes > 0:

            curr = pq.deleteMin(self.dist)
            numNodes -= 1

            for neighbor in self.network.nodes[curr].neighbors:
            
                if self.dist[neighbor.dest.node_id] > self.dist[curr] + neighbor.length:
                    self.dist[neighbor.dest.node_id] = self.dist[curr] + neighbor.length
                    self.prev[neighbor.dest.node_id] = curr
                    pq.decreaseKey(neighbor.dest.node_id, self.dist)
                    