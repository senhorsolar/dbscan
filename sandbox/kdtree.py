from collections import namedtuple
import heapq
import numpy as np
from pprint import pformat


class Node(namedtuple("Node", "point axis left right")):

    def __repr__(self):
        return pformat(tuple(self))

class KdTree:

    def __init__(self, data):

        self._size = len(data)
        self._dim = len(data[0])

        assert(all(
            len(data[i]) == self._dim for i in range(self._size)
        ))

        self._root = self._build_tree(data)


    def query(self, point, k):

        class HeapNode:

            def __init__(self, point, dist):

                self.point = point
                self.dist = dist

            def __lt__(self, other):
                return -self.dist < -other.dist

        heap = []

        def add_to_queue(point, dist):

            if len(heap) < k:
                heapq.heappush(heap, HeapNode(point, dist))
            else:
                heapq.heappushpop(heap, HeapNode(point, dist))

        def recursive_helper(node):
            if node is None:
                return

            dist = np.linalg.norm(point - node.point)
            if len(heap) < k or dist <= heap[0].dist:
                add_to_queue(node.point, dist)

            dist_to_splitting_plane = abs(node.point[node.axis] - point[node.axis])

            if point[node.axis] <= node.point[node.axis]:
                in_plane_node, out_of_plane_node = node.left, node.right
            else:
                in_plane_node, out_of_plane_node = node.right, node.left

            recursive_helper(in_plane_node)
            if dist_to_splitting_plane <= heap[0].dist:
                recursive_helper(out_of_plane_node)

        recursive_helper(self._root)
        heap.sort(key=lambda node: node.dist)
        return [heapnode.point for heapnode in heap]


    def range_query(self, point, radius):

        points = []

        def recursive_helper(node):
            if node is None:
                return

            if np.linalg.norm(point - node.point) <= radius:
                points.append(node.point)

            dist_to_splitting_plane = abs(node.point[node.axis] - point[node.axis])

            if point[node.axis] <= node.point[node.axis]:

                recursive_helper(node.left)

                if dist_to_splitting_plane < radius:
                    recursive_helper(node.right)

            else:

                recursive_helper(node.right)

                if dist_to_splitting_plane < radius:
                    recursive_helper(node.left)

        recursive_helper(self._root)
        return points

    def _build_tree(self, data):

        def recursive_helper(idxs_left, depth):
            if not idxs_left:
                return None

            axis = depth % self._dim

            idxs_left.sort(key=lambda idx: data[idx][axis])
            median_idx = len(idxs_left) // 2

            return Node(
                data[idxs_left[median_idx]],
                axis,
                recursive_helper(idxs_left[:median_idx], depth+1),
                recursive_helper(idxs_left[median_idx+1:], depth+1)
            )

        return recursive_helper(list(range(len(data))), 0)


if __name__ == '__main__':

    data = np.array(
        [
            [7, 2],
            [5, 4],
            [9, 6],
            [4, 7],
            [8, 1],
            [2, 3]
        ])
    for i, pt in enumerate(data):
        print(i, pt)

    tree = KdTree(data)
    print(tree._root)

    print(tree.range_query([7, 2], 4))
    print(tree.query([7,2], 3))
