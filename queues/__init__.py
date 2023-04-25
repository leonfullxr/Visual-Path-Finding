import heapq

# WARNING: heappush and heappop are O(log n) operations, not O(1) as FIFO should be.
class FIFOQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item):
        heapq.heappush(self._queue, (self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    def __iter__(self):
        for _, item in self._queue:
            yield item
    def __len__(self):
        return len(self._queue)

# WARNING: heappush and heappop are O(log n) operations, not O(1) as LIFO should be.    
class LIFOStack:
    def __init__(self):
        self._stack = []
        self._index = 0
    def push(self, item):
        heapq.heappush(self._stack, (-self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._stack)[-1]
    def __iter__(self):
        for _, item in self._stack:
            yield item
    def __len__(self):
        return len(self._stack)
    
class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0
    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1
    def pop(self):
        return heapq.heappop(self._queue)[-1]
    def __iter__(self):
        for _, _, item in self._queue:
            yield item
    def __len__(self):
        return len(self._queue)
    