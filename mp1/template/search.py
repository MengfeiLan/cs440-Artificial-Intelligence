# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Kelvin Ma (kelvinm2@illinois.edu) on 01/24/2021

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)


# Feel free to use the code below as you wish
# Initialize it with a list/tuple of objectives
# Call compute_mst_weight to get the weight of the MST with those objectives
# TODO: hint, you probably want to cache the MST value for sets of objectives you've already computed...
import collections
import heapq
import math

class MST:
    def __init__(self, objectives):
        self.elements = {key: None for key in objectives}

        # TODO: implement some distance between two objectives 
        # ... either compute the shortest path between them, or just use the manhattan distance between the objectives
        self.distances   = {
                (i, j): abs(i[0] - j[0]) + abs(i[1] - j[1])
                for i, j in self.cross(objectives)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)

def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """

    start = maze.start
    queue = collections.deque()
    queue.append(start)
    visited = []
    path = []
    track = {}
    end = maze.waypoints[0]

    while queue:
        x_curr, y_curr = queue.popleft()
        if (x_curr, y_curr) not in visited:
            visited.append((x_curr, y_curr))
            if (x_curr, y_curr) == end:
                break
            neighbors = maze.neighbors(x_curr, y_curr)
            for n in neighbors:
                if n not in visited and maze.navigable(n[0], n[1]):
                    queue.append(n)
                    track[n] = (x_curr, y_curr)
    while end:
        path.insert(0, end)
        end = track[end]
        if end == start:
            break
    print(visited)
    path.insert(0, end)
    return(path)

def astar_single(maze):

    start = maze.start
    queue = []
    heapq.heapify(queue)
    heapq.heappush(queue, (1, start, 0))
    visited = []
    path = []
    track = {}
    end = maze.waypoints[0]

    while queue:
        new_popped = heapq.heappop(queue)
        x_curr, y_curr = new_popped[1]
        if (x_curr, y_curr) not in visited:
            visited.append((x_curr, y_curr))
            if maze[x_curr, y_curr] in ["."]:
                break
            neighbors = maze.neighbors(x_curr, y_curr)
            for n in neighbors:
                if n not in visited and maze.navigable(n[0], n[1]):
                    heapq.heappush(queue, ( abs(n[0] - end[0]) + abs(n[1] - end[1]) + new_popped[2] + 1, n, new_popped[2] + 1))
                    track[n] = (x_curr, y_curr)
    while end:
        path.insert(0, end)
        end = track[end]
        if end == start:
            break
    path.insert(0, end)
    return path

def astar_multiple(maze):

    start = maze.start
    goals = list(maze.waypoints)
    path_all = []

    while goals != []:
        new_goal = closet_next_destination(goals, start, maze)
        next_start = new_goal
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (1, start, 0))
        visited = []
        path = []
        track = {}
        while queue:
            new_popped = heapq.heappop(queue)
            x_curr, y_curr = new_popped[1]
            if (x_curr, y_curr) not in visited:
                visited.append((x_curr, y_curr))
                if (x_curr, y_curr) == new_goal:
                    break
                neighbors = maze.neighbors(x_curr, y_curr)
                for n in neighbors:
                    if n not in visited and maze.navigable(n[0], n[1]):
                        heapq.heappush(queue, ( abs(n[0] - new_goal[0]) + abs(n[1] - new_goal[1]) + new_popped[2] + 1, n, new_popped[2] + 1))
                        track[n] = (x_curr, y_curr)
        while new_goal:
            path.insert(0, new_goal)
            new_goal = track[new_goal]
            if new_goal == start:
                break
        path.insert(0, new_goal)
        path_all.append(path)
        start = next_start
        goals.remove(next_start)

    path_final =[]
    for i in path_all:
        path_final += i[0:-1]
    path_final.append(tuple(path_all[-1][-1]))
    return path_final

def two_points_distance_astar(goal, curr, maze):
    start = curr
    queue = []
    heapq.heapify(queue)
    heapq.heappush(queue, (1, start, 0))
    visited = []
    path = []
    track = {}
    end = goal

    while queue:
        new_popped = heapq.heappop(queue)
        x_curr, y_curr = new_popped[1]
        if (x_curr, y_curr) not in visited:
            visited.append((x_curr, y_curr))
            if (x_curr, y_curr) == end:
                break
            neighbors = maze.neighbors(x_curr, y_curr)
            for n in neighbors:
                if n not in visited and maze.navigable(n[0], n[1]):
                    heapq.heappush(queue, ( abs(n[0] - end[0]) + abs(n[1] - end[1]) + new_popped[2] + 1, n, new_popped[2] + 1))
                    track[n] = (x_curr, y_curr)
    while end:
        path.insert(0, end)
        end = track[end]
        if end == start:
            break
    path.insert(0, end)
    return len(path)

def closet_next_destination(goals, curr, maze):
    dist = math.inf
    next_goal = (0, 0)
    for goal in goals:
        new_dests = list(goals)
        dist1 = two_points_distance_astar(goal, curr, maze)
        new_dests.remove(goal)
        MST_ = MST(new_dests)
        dist2 = MST_.compute_mst_weight()

        if dist1 + dist2 < dist: 
            dist = dist1 + dist2
            next_goal = goal
    return next_goal

def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    start = maze.start
    goals = list(maze.waypoints)
    path_all = []

    while goals != []:
        new_goal = closet_next_destination_2(goals, start)
        next_start = new_goal
        queue = []
        heapq.heapify(queue)
        heapq.heappush(queue, (1, start, 0))
        visited = []
        path = []
        track = {}
        while queue:
            new_popped = heapq.heappop(queue)
            x_curr, y_curr = new_popped[1]
            if (x_curr, y_curr) not in visited:
                visited.append((x_curr, y_curr))
                if (x_curr, y_curr) == new_goal:
                    break
                neighbors = maze.neighbors(x_curr, y_curr)
                for n in neighbors:
                    if n not in visited and maze.navigable(n[0], n[1]):
                        heapq.heappush(queue, ( abs(n[0] - new_goal[0]) + abs(n[1] - new_goal[1]) + new_popped[2] + 1, n, new_popped[2] + 1))
                        track[n] = (x_curr, y_curr)
        while new_goal:
            path.insert(0, new_goal)
            new_goal = track[new_goal]
            if new_goal == start:
                break
        path.insert(0, new_goal)
        path_all.append(path)
        start = next_start
        goals.remove(next_start)

    path_final =[]
    for i in path_all:
        path_final += i[0:-1]
    path_final.append(tuple(path_all[-1][-1]))
    return path_final


def closet_next_destination_2(goals, curr):
    dist = math.inf
    next_goal = (0, 0)
    for goal in goals:
        new_dests = list(goals)
        dist1 = abs(goal[0] - curr[0]) + abs(goal[1] - curr[1])
        new_dests.remove(goal)
        MST_ = MST(new_dests)
        dist2 = MST_.compute_mst_weight()

        if dist1 + dist2 < dist: 
            dist = dist1 + dist2
            next_goal = goal
    return next_goal           
