#!/usr/bin/python3

from CS312Graph import *
import time
import math

class PQArray:
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

class PQHeap:
    def __init__(self, nodes, startNode):
        self.heap = []
        self.nodePosition = []
        self.makeQueue(nodes, startNode)

    def makeQueue(self, nodes, startNode):  # O(n)
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

    def initializeNetwork( self, network ):
        assert( type(network) == CS312Graph )
        self.network = network

    def getShortestPath( self, destIndex ):
        self.dest = destIndex
        path_edges = []
        total_length = 0
        currIndex = self.dest

        while currIndex != self.source:  # Worst case: O(n)
            prevIndex = self.prev[currIndex]
            if prevIndex is not None:
                prevNode = self.network.nodes[self.prev[currIndex]]
                neighbors = prevNode.neighbors
                selectedEdge = None
                for neighbor in neighbors:
                    if neighbor.dest.node_id == currIndex:
                        selectedEdge = neighbor

                if not (selectedEdge is None):
                    path_edges.insert(0, (selectedEdge.src.loc, selectedEdge.dest.loc, '{:.0f}'.format(selectedEdge.length)))
                    total_length += selectedEdge.length
                else:
                    print("Something is wrong, line 33")

                currIndex = prevNode.node_id
            else:
                total_length = math.inf
                break

        return {'cost': total_length, 'path': path_edges}

    def computeShortestPaths( self, srcIndex, use_heap=False ):
        self.source = srcIndex
        t1 = time.time()
        self.dist = []
        self.prev = []
        self.dijkstra(self.source, use_heap)
        t2 = time.time()
        return t2 - t1
 
    def dijkstra(self, startNode, use_heap):  # Heap: O(nlog(n)), Array: O(n^2)
        numNodes = len(self.network.nodes)
        self.dist = [math.inf for _ in range(numNodes)]  # O(n)
        self.prev = [None for _ in range(numNodes)]  # O(n)

        self.dist[startNode] = 0

        if use_heap:
            priorityQueue = PQHeap(self.network.nodes, self.source)  # O(n)
        else:
            priorityQueue = PQArray(self.network.nodes)  # O(n)

        while numNodes > 0:  # O(n)
            currNode = priorityQueue.deleteMin(self.dist)  # O(log n) for heap, O(n) for array
            numNodes -= 1
            neighbors = self.network.nodes[currNode].neighbors
            for neighbor in neighbors:  # O(3)
                neighborID = neighbor.dest.node_id
                if self.dist[neighborID] > (self.dist[currNode] + neighbor.length):
                    self.dist[neighborID] = self.dist[currNode] + neighbor.length
                    self.prev[neighborID] = currNode
                    priorityQueue.decreaseKey(neighborID, self.dist)  # O(log(n)) for heap, O(1) for array