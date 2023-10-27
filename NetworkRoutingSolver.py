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
    # Time: O(1)
    def __init__(self, nodes):
        self.queue = []
        self.makeQueue(nodes)

    # Time: O(n)
    def makeQueue(self, nodes):
        for node in nodes:
            self.insert(node)  

    # Time: O(1)     
    # Space: O(n)    
    def insert(self, node):
        self.queue.append(node.node_id)  

    # Time: O(n) 
    def deleteMin(self, dist):
        if not self.queue:
            return None  # Handle the case when queue is empty
        
        # O(n) to loop through all nodes
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
    # Time: O(n)  
    # Space: O(n)
    def __init__(self, nodes, startNode):
        self.heap = []
        self.nodePosition = []
        self.makeQueue(nodes, startNode)

    # Time: O(n)
    # Space: O(n) 
    def makeQueue(self, nodes, startNode):
        self.heap = [startNode] + [node.node_id for node in nodes if node.node_id != startNode]
        self.nodePosition = [0 if node.node_id == startNode else i for i, node in enumerate(nodes, 1)]

    # Time: O(logn) 
    def insert(self, node, dist):
        self.heap.append(node.node_id)
        self.bubbleUp(node.node_id, (len(self.heap) - 1), dist)
    
    # Time: O(logn)  
    def bubbleUp(self, x, i, dist):
        while i > 0 and dist[self.heap[i//2]] > dist[x]:
            # O(1) swaps
            self.heap[i] = self.heap[i//2]
            self.heap[i//2] = x
            
            self.nodePosition[self.heap[i]], self.nodePosition[self.heap[i//2]] = self.nodePosition[self.heap[i//2]], self.nodePosition[self.heap[i]]
            
            i //= 2
    
    # Time: O(logn)
    def decreaseKey(self, node, dist):
        i = self.nodePosition[node]
        if i < len(self.heap):
            self.bubbleUp(node, i, dist)  

    # Time: O(logn) 
    def deleteMin(self, dist):   
        if not self.heap:
            return None
        
        # O(1)  
        minDist = self.heap[0]

        if len(self.heap) > 1:
            # O(logn) siftDown 
            self.heap[0] = self.heap.pop()
            self.nodePosition[self.heap[0]] = 0
            self.siftDown(self.heap[0], 0, dist)

        return minDist

    # Time: O(logn)
    def siftDown(self, x, i, dist):
        while True:
            # O(1)
            min_index = self.minChild(i, dist)
            if min_index == -1 or dist[self.heap[min_index]] >= dist[x]:
                break
            
            # O(1) swaps
            self.heap[i] = self.heap[min_index] 
            self.heap[min_index] = x

            self.nodePosition[self.heap[i]], self.nodePosition[self.heap[min_index]] = self.nodePosition[self.heap[min_index]], self.nodePosition[self.heap[i]] 

            i = min_index 
    
    # Time: O(1)
    def minChild(self, i, dist):
        first = 2*i + 1
        second = 2*i + 2
        
        if first >= len(self.heap):
            return -1
        
        if second < len(self.heap) and dist[self.heap[first]] > dist[self.heap[second]]:
            return second
        else:
            return first

class NetworkRoutingSolver:
    # Time: O(1)
    def __init__(self):
        pass

    # Time: O(1)
    def initializeNetwork(self, network):
        assert(type(network) == CS312Graph)
        self.network = network

    # Time: O(V + E), where V is the number of vertices and E is the number of edges in the path
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

    # Time: O(V^2) using PQArray, O((V + E) log V) using PQHeap, where V is the number of vertices and E is the number of edges in the network
    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time() 
        self.dijkstra(self.source, use_heap)
        t2 = time.time()
        return t2 - t1 

    # Time: O(V^2) using PQArray, O((V + E) log V) using PQHeap, where V is the number of vertices and E is the number of edges in the network
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