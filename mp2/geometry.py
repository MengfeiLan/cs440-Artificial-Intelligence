# geometry.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by James Gao (jamesjg2@illinois.edu) on 9/03/2021
# Inspired by work done by Jongdeog Lee (jlee700@illinois.edu)

"""
This file contains geometry functions necessary for solving problems in MP2
"""

import math
import numpy as np
from alien import Alien

def point_distance_to_line(p1, p2, p3):
    """
    p1, p2: locate the line
    p3: locate the point
    """
    x_diff = p2[0] - p1[0]
    y_diff = p2[1] - p1[1]
    num = abs(y_diff*p3[0] - x_diff*p3[1] + p2[0]*p1[1] - p2[1]*p1[0])
    den = math.sqrt(y_diff**2 + x_diff**2)
    return num / den

def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) < min(x2,y2)

def does_alien_touch_wall(alien, walls,granularity):
    """Determine whether the alien touches a wall

        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            walls (list): List of endpoints of line segments that comprise the walls in the maze in the format [(startx, starty, endx, endx), ...]
            granularity (int): The granularity of the map

        Return:
            True if touched, False if not
    """
    if alien.is_circle() == True:
        centroid = alien.get_centroid()
        radius = alien.get_width()
        for i in walls:
            x1, y1, x3, y3 = i
            if x1 == x3 and y1 == y3:
                if math.sqrt((centroid[0] - x1)**2 + (centroid[1] - y1)**2) <= radius + granularity/math.sqrt(2) :
                    return True
                continue
            if x1 == x3 or y1 == y3:
                distance = point_distance_to_line((x1, y1), (x3, y3), centroid)
                if distance >= radius + granularity/math.sqrt(2):
                    continue
                else:
                    reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - distance **2)
                    if x1 == x3:
                        if min(y1, y3) - reset <= centroid[1] and max(y1, y3) + reset >= centroid[1]:
                            return True
                    if y1 == y3:
                        if min(x1, x3) - reset <= centroid[0] and max(x1, x3) + reset >= centroid[0]:
                            return True
                    continue
            x2 = x1
            y2 = y3
            x4 = x3
            y4 = y1
            line_distance_1 = abs(x1 - x4 + y1 - y4)
            line_distance_2 = abs(x1 - x2 + y1 - y2)
            point_to_line_distance_1 = point_distance_to_line((x1, y1), (x2, y2), centroid)
            point_to_line_distance_2 = point_distance_to_line((x3, y3), (x4, y4), centroid)
            point_to_line_distance_3 = point_distance_to_line((x2, y2), (x3, y3), centroid)
            point_to_line_distance_4 = point_distance_to_line((x4, y4), (x1, y1), centroid)
            centroid_to_point_distance_1 = math.sqrt(abs(centroid[0] - x1)** 2 + abs(centroid[1] - y1) **2)
            centroid_to_point_distance_2 = math.sqrt(abs(centroid[0] - x2)** 2 + abs(centroid[1] - y2) **2)
            centroid_to_point_distance_3 = math.sqrt(abs(centroid[0] - x3)** 2 + abs(centroid[1] - y3) **2)
            centroid_to_point_distance_4 = math.sqrt(abs(centroid[0] - x4)** 2 + abs(centroid[1] - y4) **2)

            if abs(point_to_line_distance_1 - point_to_line_distance_2) == line_distance_1 and point_to_line_distance_1 >= radius + granularity/math.sqrt(2) and point_to_line_distance_2 >= radius + granularity/math.sqrt(2):
                continue
            if abs(point_to_line_distance_3 - point_to_line_distance_4) == line_distance_2 and point_to_line_distance_3 >= radius + granularity/math.sqrt(2) and point_to_line_distance_4 >= radius + granularity/math.sqrt(2):
                continue
            if abs(point_to_line_distance_3 - point_to_line_distance_4) == line_distance_2 and abs(point_to_line_distance_1 - point_to_line_distance_2) == line_distance_1:
                if centroid_to_point_distance_1 >= radius + granularity/math.sqrt(2) and centroid_to_point_distance_2 >= radius + granularity/math.sqrt(2) and centroid_to_point_distance_3 >= radius + granularity/math.sqrt(2) and centroid_to_point_distance_4 >= radius + granularity/math.sqrt(2):
                    continue
            else:
                return True
        

    if alien.is_circle() == False:
        head, tail = alien.get_head_and_tail()
        radius = alien.get_width()
        length = alien.get_length()
        centroid = alien.get_centroid()
        for i in walls:
            x1, y1, x3, y3 = i
            if x1 == x3 and y1 == y3:
                distance = point_distance_to_line(head, tail, (x1, y1))
                if distance < radius + granularity/math.sqrt(2):
                    return True
                continue
            if x1 == x3 or y1 == y3:
                distance = point_distance_to_line((x1, y1), (x3, y3), centroid)
                if distance > length/2 + radius + granularity/math.sqrt(2):
                    continue
                elif radius + granularity/math.sqrt(2) < distance and ((head[0] == tail[0] and x1 == x3) or (head[1] == tail[1] and y1 == y3)):
                    continue
                elif head[0] == tail[0] and x1 == x3:
                    reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - distance **2)
                    if is_overlapping(min(y1, y3) - reset, max(y1, y3) + reset, min(head[1], tail[1]), max(head[1], tail[1])):
                        return True
                    continue 
                elif head[1] == tail[1] and y1 == y3:
                    reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - distance **2)
                    if is_overlapping(min(x1, x3) - reset, max(x1, x3) + reset, min(head[0], tail[0]), max(head[0], tail[0])):
                        return True
                    continue
                elif (head[0] == tail[0] and y1 == y3) or (head[1] == tail[1] and x1 == x3):
                    if distance <= length * 0.5 + radius + granularity/math.sqrt(2):
                        if y1 == y3:
                            if is_overlapping(head[0] - radius - granularity/math.sqrt(2), head[0] + radius + granularity/math.sqrt(2), min(x1, x3), max(x1, x3)):
                                return True
                            continue
                        if x1 == x3:
                            if is_overlapping(head[1] - radius - granularity/math.sqrt(2), head[1] + radius + granularity/math.sqrt(2), min(y1, y3), max(y1, y3)):
                                return True
                            continue
                    else:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - (distance - length * 0.5) **2)
                        if y1 == y3:
                            if is_overlapping(head[0] - reset, head[0] + reset, x1, x3):
                                return True
                            continue
                        if x1 == x3:
                            if is_overlapping(head[1] - reset, head[1] + reset, y1, y3):
                                return True
                            continue

                else:
                    distance_1 = point_distance_to_line(head, tail, (x1, y1))
                    distance_2 = point_distance_to_line(head, tail, (x3, y3))
                    if distance_1 + distance_2 <= abs(x1 - x3 + y1 - y3) + 2 * (radius + granularity/math.sqrt(2)):
                        return True
                    continue
                continue
            x2 = x1
            y2 = y3
            x4 = x3
            y4 = y1

            d1 = point_distance_to_line((x1, y1), (x2, y2), head)
            d2 = point_distance_to_line((x3, y3), (x4, y4), head)
            d3 = point_distance_to_line((x1, y1), (x4, y4), head)
            d4 = point_distance_to_line((x2, y2), (x3, y3), head)
            d5 = point_distance_to_line((x1, y1), (x2, y2), tail)
            d6 = point_distance_to_line((x3, y3), (x4, y4), tail)
            d7 = point_distance_to_line((x1, y1), (x4, y4), tail)
            d8 = point_distance_to_line((x2, y2), (x3, y3), tail)

            p1 = math.sqrt((head[0] - x1)** 2 + (head[1] - y1) **2)
            p2 = math.sqrt((head[0] - x2)** 2 + (head[1] - y2) **2)
            p3 = math.sqrt((head[0] - x3)** 2 + (head[1] - y3) **2)
            p4 = math.sqrt((head[0] - x4)** 2 + (head[1] - y4) **2)
            p5 = math.sqrt((tail[0] - x1)** 2 + (tail[1] - y1) **2)
            p6 = math.sqrt((tail[0] - x2)** 2 + (tail[1] - y2) **2)
            p7 = math.sqrt((tail[0] - x3)** 2 + (tail[1] - y3) **2)
            p8 = math.sqrt((tail[0] - x4)** 2 + (tail[1] - y4) **2)
            bound = radius + granularity/math.sqrt(2)
            if not (p1 >= bound and p2 >= bound and p3 >= bound and p4 >= bound and p5 >= bound and p6 >= bound and p7 >= bound and p8 >= bound):
                return True                
            if np.isclose(head[0], tail[0]):

                if abs(d1 - d2) == abs(x2 - x3):
                    if d1 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d1 **2)
                        if is_overlapping(min(y1, y2) - reset, max(y1, y2), min(head[1], tail[1]), max(head[1], tail[1])):
                            return True
                        if is_overlapping(min(y1, y2), max(y1, y2) + reset, min(head[1], tail[1]), max(head[1], tail[1])):
                            return True
                    if d2 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d2 **2)
                        if is_overlapping(min(y1, y2) - reset, max(y1, y2), min(head[1], tail[1]), max(head[1], tail[1])):
                            return True
                        if is_overlapping(min(y1, y2), max(y1, y2) + reset, min(head[1], tail[1]), max(head[1], tail[1])):
                            return True
                if abs(d3 - d4) == abs(y1 - y2):
                    if min(head[1], tail[1]) < min(y1, y2) and max(head[1], tail[1]) > max(y1, y2):
                        continue
                    if d3 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d3 **2)
                        if min(x1, x4) - reset > head[0] and head[0] < max(x1, x4) + reset:
                            return True
                    if d4 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d4 **2)
                        if min(x1, x4) - reset > head[0] and head[0] < max(x1, x4) + reset:

                            return True                       

                if is_overlapping(min(x1, x3), max(x1, x3), head[0] - radius - granularity/math.sqrt(2), head[0] + radius + granularity/math.sqrt(2)) and (d3 < bound or d4 < bound):
                    return True

            elif head[1] == tail[1]:
                if abs(d1 - d2) == abs(y2 - y3):
                    if d1 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d1 **2)
                        if is_overlapping(min(x1, x2) - reset, max(x1, x2), min(head[0], tail[0]), max(head[0], tail[0])):
                            return True
                        if is_overlapping(min(x1, x2), max(x1, x2) + reset, min(head[0], tail[0]), max(head[0], tail[0])):
                            return True
                    if d2 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d2 **2)
                        if is_overlapping(min(x1, x2) - reset, max(x1, x2), min(head[0], tail[0]), max(head[0], tail[0])):
                            return True
                        if is_overlapping(min(x1, x2), max(x1, x2) + reset, min(head[0], tail[0]), max(head[0], tail[0])):
                            return True

                if abs(d3 - d4) == abs(x1 - x2):
                    if min(head[0], tail[0]) < min(x1, x2) and max(head[0], tail[0]) > max(x1, x2):
                        continue
                    if d3 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d3 **2)
                        if min(y1, y4) - reset > head[1] and head[1] < max(y1, y4) + reset:
                            return True
                    if d4 < bound:
                        reset = math.sqrt((radius + granularity/math.sqrt(2))** 2 - d4 **2)
                        if min(y1, y4) - reset > head[1] and head[1] < max(y1, y4) + reset:

                            return True  
                if is_overlapping(min(x1, x2), max(x1, x2), min(head[0], tail[0]), max(head[0], tail[0])) and (d3 < bound or d4 < bound):
                    return True
                if is_overlapping(min(y1, y3), max(y1, y3), head[1] - radius - granularity/math.sqrt(2), head[1] + radius + granularity/math.sqrt(2)) and (d1 < bound or d2 < bound):
                    return True
                # if np.isclose(abs(d3 - d4), y1 - y3):
                #     continue
 
    return False

def does_alien_touch_goal(alien, goals):
    """Determine whether the alien touches a goal
        
        Args:
            alien (Alien): Instance of Alien class that will be navigating our map
            goals (list): x, y coordinate and radius of goals in the format [(x, y, r), ...]. There can be multiple goals
        
        Return:
            True if a goal is touched, False if not.
    """
    if alien.is_circle() == True:
        centroid = alien.get_centroid()
        redius = alien.get_width()
        for goal in goals:
            if math.sqrt((centroid[0] - goal[0])**2 + (centroid[1] - goal[1])**2) <= redius + goal[2]:
                return True
    if alien.is_circle() == False:
        head, tail = alien.get_head_and_tail()
        radius = alien.get_width()
        if head[0] == tail[0]:
            flag = 1
        elif head[1] == tail[1]:
            flag = 0
        for goal in goals:
            if goal[flag] <= max(head[flag], tail[flag]) and goal[flag] >= min(head[flag], tail[flag]):
                if radius + goal[2] >= point_distance_to_line(head, tail, (goal[0], goal[1])):
                    return True
            else: 
                if radius + goal[2] >= math.sqrt((goal[0] - head[0])**2 + (goal[1] - head[1])**2) or radius + goal[2] >= math.sqrt((goal[0] - tail[0])**2 + (goal[1] - tail[1])**2):
                    return True
    return False

def is_alien_within_window(alien, window,granularity):
    """Determine whether the alien stays within the window
        
        Args:
            alien (Alien): Alien instance
            window (tuple): (width, height) of the window
            granularity (int): The granularity of the map
    """
    head, tail = alien.get_head_and_tail()
    radius = alien.get_width()
    if (head[0] + radius + granularity/math.sqrt(2) >= window[0]) or (tail[0] + radius + granularity/math.sqrt(2) >= window[0]) or (head[1] + radius + granularity/math.sqrt(2) >= window[1]) or (tail[1] + radius + granularity/math.sqrt(2) >= window[1]):
        return False
    if (head[0] - radius - granularity/math.sqrt(2) <= 0) or (tail[0] - radius - granularity/math.sqrt(2) <= 0) or (head[1] - radius - granularity/math.sqrt(2) <= 0) or (tail[1] - radius - granularity/math.sqrt(2) <= 0):
        return False
    return True

if __name__ == '__main__':
    #Walls, goals, and aliens taken from Test1 map
    walls =   [(0,100,100,100),  
                (0,140,100,140),
                (100,100,140,110),
                (100,140,140,130),
                (140,110,175,70),
                (140,130,200,130),
                (200,130,200,10),
                (200,10,140,10),
                (175,70,140,70),
                (140,70,130,55),
                (140,10,130,25),
                (130,55,90,55),
                (130,25,90,25),
                (90,55,90,25)]
    goals = [(110, 40, 10)]
    window = (220, 200)

    #Initialize Aliens and perform simple sanity check. 
    alien_ball = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Ball', window)	
    assert not does_alien_touch_wall(alien_ball, walls, 0), f'does_alien_touch_wall(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_ball, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_ball.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_ball, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_ball.get_config()} returns False, expected: True'

    alien_horz = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)	
    assert not does_alien_touch_wall(alien_horz, walls, 0), f'does_alien_touch_wall(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert not does_alien_touch_goal(alien_horz, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_horz.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_horz, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_horz.get_config()} returns False, expected: True'

    alien_vert = Alien((30,120), [40, 0, 40], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)	
    assert does_alien_touch_wall(alien_vert, walls, 0),f'does_alien_touch_wall(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'
    assert not does_alien_touch_goal(alien_vert, goals), f'does_alien_touch_goal(alien, walls) with alien config {alien_vert.get_config()} returns True, expected: False'
    assert is_alien_within_window(alien_vert, window, 0), f'is_alien_within_window(alien, walls) with alien config {alien_vert.get_config()} returns False, expected: True'

    edge_horz_alien = Alien((50, 100), [100, 0, 100], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Horizontal', window)
    edge_vert_alien = Alien((200, 70), [120, 0, 120], [11, 25, 11], ('Horizontal','Ball','Vertical'), 'Vertical', window)

    def test_helper(alien : Alien, position, truths):
        alien.set_alien_pos(position)
        config = alien.get_config()
        assert does_alien_touch_wall(alien, walls, 0) == truths[0], f'does_alien_touch_wall(alien, walls) with alien config {config} returns {not truths[0]}, expected: {truths[0]}'
        assert does_alien_touch_goal(alien, goals) == truths[1], f'does_alien_touch_goal(alien, goals) with alien config {config} returns {not truths[1]}, expected: {truths[1]}'
        assert is_alien_within_window(alien, window, 0) == truths[2], f'is_alien_within_window(alien, window) with alien config {config} returns {not truths[2]}, expected: {truths[2]}'

    alien_positions = [
                        #Sanity Check
                        (0, 100),

                        #Testing window boundary checks
                        (25.6, 25.6),
                        (25.5, 25.5),
                        (194.4, 174.4),
                        (194.5, 174.5),

                        #Testing wall collisions
                        (30, 112),
                        (30, 113),
                        (30, 105.5),
                        (30, 105.6), # Very close edge case
                        (30, 135),
                        (140, 120),
                        (187.5, 70), # Another very close corner case, right on corner
                        
                        #Testing goal collisions
                        (110, 40),
                        (145.5, 40), # Horizontal tangent to goal
                        (110, 62.5), # ball tangent to goal
                        
                        #Test parallel line oblong line segment and wall
                        (50, 100),
                        (200, 100),
                        (205.5, 100) #Out of bounds
                    ]

    #Truths are a list of tuples that we will compare to function calls in the form (does_alien_touch_wall, does_alien_touch_goal, is_alien_within_window)
    alien_ball_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]
    alien_horz_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, False, True),
                            (True, True, True),
                            (False, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, False),
                            (True, False, False)
                        ]
    alien_vert_truths = [
                            (True, False, False),
                            (False, False, True),
                            (False, False, False),
                            (False, False, True),
                            (False, False, False),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True),
                            (False, False, True),
                            (True, True, True),
                            (False, False, True),
                            (True, True, True),
                            (True, False, True),
                            (True, False, True),
                            (True, False, True)
                        ]

    for i in range(len(alien_positions)):
        test_helper(alien_ball, alien_positions[i], alien_ball_truths[i])
        test_helper(alien_horz, alien_positions[i], alien_horz_truths[i])
        test_helper(alien_vert, alien_positions[i], alien_vert_truths[i])

    #Edge case coincide line endpoints
    test_helper(edge_horz_alien, edge_horz_alien.get_centroid(), (True, False, False))
    test_helper(edge_horz_alien, (110,55), (True, True, True))
    test_helper(edge_vert_alien, edge_vert_alien.get_centroid(), (True, False, True))


    print("Geometry tests passed\n")