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
# searchMethod is the search method specified by --method flag (bfs,astar)
# You may need to slight change your previous search functions in MP1 since this is 3-d maze

from collections import deque
from heapq import heappop, heappush

import heapq
import queue
import copy
from util import *

def search(maze, searchMethod):
    return {
        "bfs": bfs,
    }.get(searchMethod, [])(maze)

def getNeighbors(maze, alpha, beta):
    """
    This function returns the valid neighbors in the maze for a given (alpha, beta)
    """
    alpha_idx, beta_idx = angleToIdx((alpha, beta), maze.offsets, maze.granularity)
    neighbors = [(alpha_idx, beta_idx-1), (alpha_idx-1, beta_idx), (alpha_idx, beta_idx+1), (alpha_idx+1, beta_idx)]
    for i in neighbors:
        neighbor_alpha, neighbor_beta = idxToAngle((i[0], i[1]), maze.offsets, maze.granularity)
        if not maze.isValidMove(neighbor_alpha, neighbor_beta):
            neighbors.remove(i)

    return neighbors

def bfs(maze):
    # Write your code here
    """
    This function returns optimal path in a list, which contains start and objective.
    If no path found, return None. 
    """
    start = maze.getStart()
    start_idx = angleToIdx(start, maze.offsets, maze.granularity)
    visited = []
    visited.append(start)
    q = queue.Queue()
    q.put(start)
    path = []

    rows, cols = maze.getDimensions()
    # previous = [[0 for i in range(cols)] for j in range(rows)]
    previous = {}
    previous_state = angleToIdx(maze.getObjectives()[0], maze.offsets, maze.granularity)

    while not q.empty():
        current = q.get()
        current_idx = angleToIdx(current, maze.offsets, maze.granularity)
        if maze.isObjective(current[0], current[1]):
            previous_state = current_idx
            break

        nbhrs = maze.getNeighbors(current[0], current[1])
        for i in nbhrs:
            nbhr_idx = angleToIdx(i, maze.offsets, maze.granularity)
            if (not (i in visited)):
                q.put(i)
                visited.append(i)
                previous[nbhr_idx] = current_idx

    if q.empty():
        return None

    while previous_state != start_idx:
        path.append(previous_state)
        previous_state = previous[previous_state]
    path.append(start_idx)

    path_angle = []
    for i in path:
        path_angle.append(idxToAngle(i, maze.offsets, maze.granularity))

    return path_angle[::-1]
