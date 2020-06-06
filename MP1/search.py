# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018

"""
This is the main entry point for MP1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,extra)

import heapq
import queue
import copy
from copy import deepcopy

def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "extra": extra,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    visited = []
    visited.append(start)
    q = queue.Queue()
    q.put(start)
    path = []

    rows, cols = maze.getDimensions()
    previous = [[0 for i in range(cols)] for j in range(rows)]

    while not q.empty():
        current = q.get()
        if maze.isObjective(current[0], current[1]):
            break

        nbhrs = maze.getNeighbors(current[0], current[1])
        for i in nbhrs:
            if (not (i in visited)):
                q.put(i)
                visited.append(i)
                previous[i[0]][i[1]] = current

    previous_state = maze.getObjectives()[0]

    while previous_state != start:
        path.append(previous_state)
        previous_state = previous[previous_state[0]][previous_state[1]]
    path.append(start)

    return path[::-1]

def manhattan(begin, end):
    return abs(begin[0]-end[0]) + abs(begin[1]-end[1])

def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    goal = maze.getObjectives()
    rows, cols = maze.getDimensions()
    # previous = [[0 for i in range(cols)] for j in range(rows)]
    previous = [[0]*cols for i in range(rows)]

    visited = []
    path = []
    frontier = []  # frontier list, which is a priority queue
    heapq.heappush(frontier, (0, 0, start))  # f, g, coordinate

    while len(frontier) > 0:
        curr_tuple = heapq.heappop(frontier)
        curr_coordinate = curr_tuple[2]
        curr_g = curr_tuple[1]
        visited.append(curr_coordinate)

        if curr_coordinate == goal[0]:
            break


        else:
            neighbors = maze.getNeighbors(curr_coordinate[0], curr_coordinate[1])
            for i in neighbors:

                if(not(i in visited)):
                    neighbor_g = 1 + curr_g
                    neighbor_h = manhattan(i, goal[0])
                    neighbor_f = neighbor_g + neighbor_h
                    in_frontier = 0
                    for x in frontier:
                        if i == x[2]:  # neighbor is already in the frontier list
                            in_frontier = 1
                            if neighbor_f < x[0]:
                                frontier.remove(x)
                                heapq.heappush(frontier, (neighbor_f, neighbor_g, i))
                                previous[i[0]][i[1]] = curr_coordinate

                    if in_frontier == 0:
                        heapq.heappush(frontier, (neighbor_f, neighbor_g, i))
                        previous[i[0]][i[1]] = curr_coordinate

    previous_state = goal[0]

    while previous_state != start:
        path.append(previous_state)
        previous_state = previous[previous_state[0]][previous_state[1]]

    path.append(start)

    return path[::-1]


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.
        
    @param maze: The maze to execute the search on.
        
    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    # TODO: Write your code here
    start = maze.getStart()
    goal = maze.getObjectives()
    previous = {}

    visited = []
    path = []
    frontier = []                                     # frontier list, which is a priority queue
    heapq.heappush(frontier, (0, 0, start, goal))     # f, g, coordinate, goals' coordinates
    last_goal = copy.deepcopy(goal[0])                               # default value for the last goal reached
    state_number = 0

    while len(frontier) > 0:
        state_number = state_number + 1
        curr_tuple = heapq.heappop(frontier)
        curr_coordinate = curr_tuple[2]
        curr_g = curr_tuple[1]
        curr_goals = copy.deepcopy(curr_tuple[3])
        visited.append((curr_coordinate, curr_goals))

        is_goal = 0

        if curr_coordinate in curr_goals:             # if we hit a goal
            if len(curr_goals) == 1:                  # if we hit the last remaining goal
                last_goal = curr_coordinate           # set the last goal visited
                break
            else:                                     # if an unvisited goal
                is_goal = 1

        neighbors = maze.getNeighbors(curr_coordinate[0], curr_coordinate[1])
        for i in neighbors:
            is_neighbor_visited = 0
            neighbor_goals = copy.deepcopy(curr_goals)
            if is_goal == 1:                            # if current coordinate is a goal
                neighbor_goals.remove(curr_coordinate)  # remove current coordinate from its neighbor's goal list

            if (i, neighbor_goals) in visited:
                is_neighbor_visited = 1

            if is_neighbor_visited == 0:                # if the state is not visited before
                neighbor_g = 1 + curr_g
                neighbor_h = 0
                for j in neighbor_goals:
                    neighbor_h = manhattan(i, j)

                neighbor_f = neighbor_g + neighbor_h

                in_frontier = 0
                for x in frontier:
                    if neighbor_goals == x[3]:          # if both the coordinate and goals left are the same
                        if i == x[2]:
                            in_frontier = 1             # the neighbor is already in the frontier list
                            if neighbor_f < x[0]:
                                frontier.remove(x)
                                heapq.heappush(frontier, (neighbor_f, neighbor_g, i, neighbor_goals))
                                previous[i, tuple(neighbor_goals)] = (curr_coordinate, tuple(curr_goals))

                if in_frontier == 0:
                    heapq.heappush(frontier, (neighbor_f, neighbor_g, i, neighbor_goals))
                    previous[i, tuple(neighbor_goals)] = (curr_coordinate, tuple(curr_goals))

    previous_state = (last_goal, (last_goal,))

    while previous_state != (start, tuple(goal)):
        path.append(previous_state[0])
        previous_state = previous[previous_state[0], previous_state[1]]

    path.append(start)

    return path[::-1]

def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here


    return []


def extra(maze):
    """
    Runs extra credit suggestion.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    return []
