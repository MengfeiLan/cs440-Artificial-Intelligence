# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains search functions.
"""
# Search should return the path and the number of states explored.
# The path should be a list of tuples in the form (alpha, beta, gamma) that correspond
# to the positions of the path taken by your search algorithm.
# Number of states explored should be a number.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush
import collections

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def bfs(maze, ispart1=False):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 

    Args:
        maze: Maze instance from maze.py
        ispart1: pass this variable when you use functions such as getNeighbors and isObjective. DO NOT MODIFY THIS
    """

    start = maze.getStart()
    queue = collections.deque()
    queue.append(start)
    visited = []
    path = []
    track = {}
    end = maze.getObjectives()
    end_ = []
    while queue:
        x_curr, y_curr, z_curr = queue.popleft()
        if (x_curr, y_curr, z_curr) not in visited:
            visited.append((x_curr, y_curr, z_curr))
            if (x_curr, y_curr, z_curr) in end:
                end_ = (x_curr, y_curr, z_curr)
                break
            neighbors = maze.getNeighbors(x_curr, y_curr, z_curr, ispart1)
            for n in neighbors:
                if n not in visited and maze.isValidMove(n[0], n[1], n[2], ispart1):
                    queue.append(n)
                    track[n] = (x_curr, y_curr, z_curr)
    
    while end_:
        path.insert(0, end_)
        end_ = track[end_]
        if end_ == start:
            break
            
    if end_ != start:
        return None

    path.insert(0, end_)
    if path != []:
        return path
    else: 
        return None
