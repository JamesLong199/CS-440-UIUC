# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains geometry functions that relate with Part1 in MP2.
"""

import math
import numpy as np
from const import *

def computeCoordinate(start, length, angle):
    """Compute the end cooridinate based on the given start position, length and angle.

        Args:
            start (tuple): base of the arm link. (x-coordinate, y-coordinate)
            length (int): length of the arm link
            angle (int): degree of the arm link from x-axis to counter-clockwise

        Return:
            End position (int,int): of the arm link, (x-coordinate, y-coordinate)
    """
    end_x = start[0] + int(length * math.cos(math.radians(angle)))
    end_y = start[1] - int(length * math.sin(math.radians(angle)))

    return end_x, end_y

def circle_line_intersect(start, end, center, radius):
    """

    :param start: coordinate (x,y) -- start point of the line
    :param end: coordinate (x,y) -- end point of the line
    :param center: coordinate (x,y) -- center of the circle
    :param radius: radius -- center of the circle
    :return: true if there is intersection(s) between given line and circle, false otherwise

    """

    start_ = np.array(start)
    end_ = np.array(end)
    vector = end_ - start_
    object_ = np.array(center)

    a = np.dot(vector, vector)
    b = 2 * np.dot(vector, start_ - object_)
    c = np.dot(start_, start_) + np.dot(object_, object_) - 2 * np.dot(start_, object_) - radius ** 2

    delta = b ** 2 - 4 * a * c

    if delta < 0:
        return False

    t1 = (-b + math.sqrt(delta)) / (2 * a)
    t2 = (-b - math.sqrt(delta)) / (2 * a)

    if 0 <= t1 <= 1 or 0 <= t2 <= 1:
        return True


def doesArmTouchObjects(armPosDist, objects, isGoal=False):
    """Determine whether the given arm links touch any obstacle or goal

        Args:
            armPosDist (list): start and end position and padding distance of all arm links [(start, end, distance)]
            objects (list): x-, y- coordinate and radius of object (obstacles or goals) [(x, y, r)]
            isGoal (bool): True if the object is a goal and False if the object is an obstacle.
                           When the object is an obstacle, consider padding distance.
                           When the object is a goal, no need to consider padding distance.
        Return:
            True if touched. False if not.
    """

    for arm in armPosDist:
        for obj in objects:
            eff_radius = obj[2]
            if not isGoal:
                eff_radius = eff_radius + arm[2]
            if circle_line_intersect(arm[0], arm[1], (obj[0], obj[1]), eff_radius):
                return True

    return False




def doesArmTipTouchGoals(armEnd, goals):
    """Determine whether the given arm tip touch goals

        Args:
            armEnd (tuple): the arm tip position, (x-coordinate, y-coordinate)
            goals (list): x-, y- coordinate and radius of goals [(x, y, r)]. There can be more than one goal.
        Return:
            True if arm tick touches any goal. False if not.
    """

    for goal in goals:
        distance = math.sqrt(math.pow((armEnd[0] - goal[0]), 2) + math.pow((armEnd[1] - goal[1]), 2))
        # distance from arm tip to center of goal

        if distance <= goal[2]:
            return True

    return False


def isArmWithinWindow(armPos, window):
    """Determine whether the given arm stays in the window

        Args:
            armPos (list): start and end positions of all arm links [(start, end)]
            window (tuple): (width, height) of the window

        Return:
            True if all parts are in the window. False if not.
    """

    for arm in armPos:

        if arm[0][0] < 0 or arm[0][0] > window[0] or arm[0][1] < 0 or arm[0][1] > window[1]:
            return False
        if arm[1][0] < 0 or arm[1][0] > window[0] or arm[1][1] < 0 or arm[1][1] > window[1]:
            return False

    return True


if __name__ == '__main__':
    computeCoordinateParameters = [((150, 190),100,20), ((150, 190),100,40), ((150, 190),100,60), ((150, 190),100,160)]
    resultComputeCoordinate = [(243, 156), (226, 126), (200, 104), (57, 156)]
    testRestuls = [computeCoordinate(start, length, angle) for start, length, angle in computeCoordinateParameters]
    assert testRestuls == resultComputeCoordinate

    testArmPosDists = [((100,100), (135, 110), 4), ((135, 110), (150, 150), 5)]
    testObstacles = [[(120, 100, 5)], [(110, 110, 20)], [(160, 160, 5)], [(130, 105, 10)]]
    resultDoesArmTouchObjects = [
        True, True, False, True, False, True, False, True,
        False, True, False, True, False, False, False, True
    ]

    testResults = []
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle))

    print("\n")
    for testArmPosDist in testArmPosDists:
        for testObstacle in testObstacles:
            testResults.append(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))
            # print(testArmPosDist)
            # print(doesArmTouchObjects([testArmPosDist], testObstacle, isGoal=True))

    assert resultDoesArmTouchObjects == testResults

    testArmEnds = [(100, 100), (95, 95), (90, 90)]
    testGoal = [(100, 100, 10)]
    resultDoesArmTouchGoals = [True, True, False]

    testResults = [doesArmTipTouchGoals(testArmEnd, testGoal) for testArmEnd in testArmEnds]
    assert resultDoesArmTouchGoals == testResults

    testArmPoss = [((100,100), (135, 110)), ((135, 110), (150, 150))]
    testWindows = [(160, 130), (130, 170), (200, 200)]
    resultIsArmWithinWindow = [True, False, True, False, False, True]
    testResults = []
    for testArmPos in testArmPoss:
        for testWindow in testWindows:
            testResults.append(isArmWithinWindow([testArmPos], testWindow))
    assert resultIsArmWithinWindow == testResults

    print("Test passed\n")
