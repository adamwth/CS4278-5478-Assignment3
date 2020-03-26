#!/usr/bin/env python
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse
import heapq as pq
import numpy as np
import json

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


def dump_action_table(action_table, filename):
    """dump the MDP policy into a json file

    Arguments:
        action_table {dict} -- your mdp action table. It should be of form {'1,2,0': (1, 0), ...}
        filename {str} -- output filename
    """
    tab = dict()
    for k, v in action_table.items():
        key = [str(i) for i in k]
        key = ','.join(key)
        tab[key] = v

    with open(filename, 'w') as fout:
        json.dump(tab, fout)


class Planner:
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05. 
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        rospy.init_node('planner')
        self.map = None
        self.pose = None
        self.goal = None
        self.path = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
        self.inflation_ratio = inflation_ratio
        self.map_callback()
        self.sb_obs = rospy.Subscriber('/scan', LaserScan, self._obs_callback)
        self.sb_pose = rospy.Subscriber(
            '/base_pose_ground_truth', Odometry, self._pose_callback)
        self.sb_goal = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self._goal_callback)
        self.controller = rospy.Publisher(
            '/mobile_base/commands/velocity', Twist, queue_size=10)
        rospy.sleep(1)

    def map_callback(self):
        """Get the occupancy grid and inflate the obstacle by some pixels. You should implement the obstacle inflation yourself to handle uncertainty.
        """
        self.map = rospy.wait_for_message('/map', OccupancyGrid).data

        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask
        self.map = np.reshape(self.map, (self.world_height, self.world_width))
        # print(self.map)
        # print("===============")

        # you should inflate the map to get self.aug_map
        self.aug_map = copy.deepcopy(self.map)
        obstacles = []
        rows = self.aug_map.shape[0]
        cols = self.aug_map.shape[1]

        # find obstacle pixels
        for i in range(0, cols):
            for j in range(0, rows):
                if self.aug_map[j, i] == 100:
                    obstacles.append((i, j))

        # for i in obstacles:
        #     print((i[0] * self.resolution, i[1] * self.resolution))
        
        # inflate those found obstacle pixels only
        inflation = int(round(self.inflation_ratio))
        for obstacle in obstacles:
            x = obstacle[0]
            y = obstacle[1]
            for inflate in range(-inflation, inflation + 1):
                new_x = x + inflate
                new_y = y + inflate
                if new_x >= 0 and new_x < self.world_width:
                    self.aug_map[y, new_x] = 100
                if new_y >= 0 and new_y < self.world_height:
                    self.aug_map[new_y, x] = 100
                if new_x >= 0 and new_x < self.world_width \
                    and new_y >= 0 and new_y < self.world_height:
                    self.aug_map[new_y, new_x] = 100
                
        # for i in range(0, cols):
        #     for j in range(0, rows):
        #         if self.aug_map[j, i] != -1:
        #             if i * self.resolution > 4.5 and i * self.resolution < 5.5:
        #                 print((i * self.resolution, j * self.resolution))
        print(self.aug_map)

    def _pose_callback(self, msg):
        """get the raw pose of the robot from ROS

        Arguments:
            msg {Odometry} -- pose of the robot from ROS
        """
        self.pose = msg

    def _goal_callback(self, msg):
        self.goal = msg
        self.generate_plan()

    def _get_goal_position(self):
        goal_position = self.goal.pose.position
        return (goal_position.x, goal_position.y)

    def set_goal(self, x, y, theta=0):
        """set the goal of the planner

        Arguments:
            x {int} -- x of the goal
            y {int} -- y of the goal

        Keyword Arguments:
            theta {int} -- orientation of the goal; we don't consider it in our planner (default: {0})
        """
        a = PoseStamped()
        a.pose.position.x = x
        a.pose.position.y = y
        a.pose.orientation.z = theta
        self.goal = a

    def _obs_callback(self, msg):
        """get the observation from ROS; currently not used in our planner; researve for the next assignment

        Arguments:
            msg {LaserScan} -- LaserScan ROS msg for observations
        """
        self.last_obs = msg

    def _d_from_goal(self, pose):
        """compute the distance from current pose to the goal; only for goal checking

        Arguments:
            pose {list} -- robot pose

        Returns:
            float -- distance to the goal
        """
        goal = self._get_goal_position()
        return sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)

    def _check_goal(self, pose):
        """Simple goal checking criteria, which only requires the current position is less than 0.25 from the goal position. The orientation is ignored

        Arguments:
            pose {list} -- robot post

        Returns:
            bool -- goal or not
        """
        if self._d_from_goal(pose) < 0.25:
            return True
        else:
            return False

    def create_control_msg(self, x, y, z, ax, ay, az):
        """a wrapper to generate control message for the robot.

        Arguments:
            x {float} -- vx
            y {float} -- vy
            z {float} -- vz
            ax {float} -- angular vx
            ay {float} -- angular vy
            az {float} -- angular vz

        Returns:
            Twist -- control message
        """
        message = Twist()
        message.linear.x = x
        message.linear.y = y
        message.linear.z = z
        message.angular.x = ax
        message.angular.y = ay
        message.angular.z = az
        return message

    class Node:
        def __init__(x, y, theta, f, g, parent):
            self.x = x
            self.y = y
            self.theta = theta
            self.f = f
            self.g = g
            self.parent = parent

        def __cmp__(self, other):
            return cmp(self.f, other.f)

    def generate_plan(self):
        """TODO: FILL ME! This function generates the plan for the robot, given a goal.
        You should store the list of actions into self.action_seq.

        In discrete case (task 1 and task 3), the robot has only 4 heading directions
        0: east, 1: north, 2: west, 3: south

        Each action could be: (1, 0) FORWARD, (0, 1) LEFT 90 degree, (0, -1) RIGHT 90 degree

        In continuous case (task 2), the robot can have arbitrary orientations

        Each action could be: (v, \omega) where v is the linear velocity and \omega is the angular velocity
        """
        FORWARD = (1, 0)
        LEFT = (0, 1)
        RIGHT = (0, -1)
        actions = [FORWARD, LEFT, RIGHT]

        self.action_seq = []
        visited = set()
        action_from_parent = {}
        heap = []

        curr_state = self.get_current_discrete_state()
        g = 0
        h = self.get_heuristic_cost(curr_state)
        f = g + h
        
        curr_state_cost = (f, g) + curr_state
        pq.heappush(heap, curr_state_cost)
        action_from_parent[curr_state] = None

        last_state = None
        # print(self.aug_map[40, 60])
        while len(heap) > 0:
            curr_state_cost = pq.heappop(heap)
            f = curr_state_cost[0]
            g = curr_state_cost[1]
            x = curr_state_cost[2]
            y = curr_state_cost[3]
            theta = curr_state_cost[4]
            # print("f: %f, g: %f, x: %d, y: %d, theta: %d" % (f, g, x, y, theta))
            curr_state = (x, y, theta)
            if curr_state in visited:
                continue
                # print("this state is already visited (%d, %d, %d)" % (x, y, theta))
            # print("f: %f, g: %f, x: %d, y: %d, theta: %d" % (f, g, x, y, theta))
            visited.add(curr_state)
            if self._check_goal(curr_state):
                last_state = curr_state
                break
            for action in actions:
                v = action[0]
                w = action[1]
                new_state = self.discrete_motion_predict(x, y, theta, v, w)
                if new_state is None:
                    # print("can't go there")
                    continue

                new_x = new_state[0]
                new_y = new_state[1]

                if new_state in visited:
                    # print("already visited new state: (%d, %d, %d)" % (new_x, new_y, new_state[2]))
                    continue

                # new_g = g
                # if new_x != x or new_y != y:
                #     new_g += 1
                new_g = g + 1
                new_h = self.get_heuristic_cost(new_state)
                new_f = new_g + new_h
                new_state_cost = (new_f, new_g) + new_state
                pq.heappush(heap, new_state_cost)

                if new_state not in action_from_parent:
                    action_from_parent[new_state] = (curr_state, action, new_f)
                else:
                    old_f = action_from_parent[new_state][2]
                    if new_f < old_f:
                        action_from_parent[new_state] = (curr_state, action, new_f)
            # next_state_cost = heap[0]
            # action = self.get_action(curr_state_cost, next_state_cost)
            # self.action_seq += action

            # for new_pos in [(0, 1, 1), (0, -1, 3), (-1, 0, 2), (1, 0, 0)]:
            #     new_state = (x + new_pos[0], y + new_pos[1], new_pos[2])
            #     g = self.get_map_cost(new_state) + curr_state[1]
            #     h = self.get_heuristic_cost(new_state)
            #     f = g + h
            #     new_state_cost = (f, g) + new_state
            #     pq.heappush(heap, new_state_cost)
            # next_state_cost = heap[0]
            # action = self.get_action(curr_state_cost, next_state_cost)
            # self.action_seq += action
            # up = (curr_state[0], curr_state[1] + 1, 1)
            # down = (curr_state[0], curr_state[1] - 1, 3)
            # left = (curr_state[0] - 1, curr_state[1], 2)
            # right = (curr_state[0] + 1, curr_state[1], 0)
        # for k, v in sorted(action_from_parent.iteritems()):
        #     print (k, v)
        curr = last_state
        print("====")
        print("START BACKTRACKING")
        print(curr)
        while True:
            afp = action_from_parent[curr]
            if afp is None:
                break
            print(afp)
            parent = afp[0]
            action = afp[1]
            self.action_seq.insert(0, action)
            curr = parent
        # print(self.action_seq)



    # def get_action(self, curr_state, next_state):
    #     diff_theta = next_state[4] - curr_state[4]
    #     rotate_right = (0, -1)
    #     rotate_left = (0, 1)
    #     actions = []
    #     if diff_theta == 1 or diff_theta == -3:
    #         actions.append(rotate_right)
    #     elif diff_theta == 2 or diff_theta == -2:
    #         actions.append(rotate_right)
    #         actions.append(rotate_right)
    #     elif diff_theta == -1 or diff_theta == 3:
    #         actions.append(rotate_left)
    #     else:
    #         return []
    #     forward = (0, 1)
    #     actions.append(forward)
    #     return actions

    # def get_map_cost(self, state):
    #     x = state[0]
    #     y = state[1]
    #     return self.aug_map[x, y]

    def get_heuristic_cost(self, state):
        x = state[0]
        y = state[1]
        return self._d_from_goal((x, y))

    def get_current_continuous_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot
        """
        x = self.pose.pose.pose.position.x
        y = self.pose.pose.pose.position.y
        orientation = self.pose.pose.pose.orientation
        ori = [orientation.x, orientation.y, orientation.z,
               orientation.w]

        phi = np.arctan2(2 * (ori[0] * ori[1] + ori[2] * ori[3]), 1 - 2 *
                         (ori[1] ** 2 + ori[2] ** 2))
        return (x, y, phi)

    def get_current_discrete_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot in discrete space, e.g., (1, 1, 1) where the robot is facing north
        """
        x, y, phi = self.get_current_continuous_state()
        def rd(x): return int(round(x))
        return rd(x), rd(y), rd(phi / (np.pi / 2))

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size
        
        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
        
        Returns:
            bool -- True for collision, False for non-collision
        """
        map_x = int(round(x / self.resolution))
        map_y = int(round(y / self.resolution))
        if map_x < 0 or map_x >= self.world_width:
            # print("collision at: (%f, %f)" % (x, y))
            return True
        if map_y < 0 or map_y >= self.world_height:
            # print("collision at: (%f, %f)" % (x, y))
            return True
        if self.aug_map[map_y, map_x] == 100:
            # print("collision at: (%f, %f)" % (x, y))
            return True

        # print("NO COLLISION AT: (%f, %f)" % (x, y))
        return False

    def motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """predict the next pose of the robot given controls. Returns None if the robot collide with the wall
        The robot dynamics are provided in the homework description

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
            theta {float} -- current theta of robot
            v {float} -- linear velocity 
            w {float} -- angular velocity

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        num_steps = int(dt * frequency)
        dx = 0
        dy = 0
        for i in range(num_steps):
            if w != 0:
                dx = - v / w * np.sin(theta) + v / w * \
                    np.sin(theta + w / frequency)
                dy = v / w * np.cos(theta) - v / w * \
                    np.cos(theta + w / frequency)
            else:
                dx = v*np.cos(theta)/frequency
                dy = v*np.sin(theta)/frequency
            x += dx
            y += dy

            if self.collision_checker(x, y):
                return None
            theta += w / frequency
        return x, y, theta

    def discrete_motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """discrete version of the motion predict. Note that since the ROS simulation interval is set to be 0.5 sec
        and the robot has a limited angular speed, to achieve 90 degree turns, we have to execute two discrete actions
        consecutively. This function wraps the discrete motion predict.

        Please use it for your discrete planner.

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot
            theta {int} -- current theta of robot
            v {int} -- linear velocity
            w {int} -- angular velocity (0, 1, 2, 3)

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        w_radian = w * np.pi/2
        first_step = self.motion_predict(x, y, theta*np.pi/2, v, w_radian)
        if first_step:
            second_step = self.motion_predict(
                first_step[0], first_step[1], first_step[2], v, w_radian)
            if second_step:
                return (round(second_step[0]), round(second_step[1]), round(second_step[2] / (np.pi / 2)) % 4)
        return None

    def publish_control(self):
        """publish the continuous controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1])
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_discrete_control(self):
        """publish the discrete controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(
                action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)

    def publish_stochastic_control(self):
        """publish stochastic controls in MDP. 
        In MDP, we simulate the stochastic dynamics of the robot as described in the assignment description.
        Please use this function to publish your controls in task 3, MDP. DO NOT CHANGE THE PARAMETERS :)
        We will test your policy using the same function.
        """
        current_state = self.get_current_state()
        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            current_state = self.get_current_state()
            action = self.action_table[current_state[0],
                                       current_state[1], current_state[2] % 4]
            if action == (1, 0):
                r = np.random.rand()
                if r < 0.9:
                    action = (1, 0)
                elif r < 0.95:
                    action = (np.pi/2, 1)
                else:
                    action = (np.pi/2, -1)
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            time.sleep(1)
            current_state = self.get_current_state()


if __name__ == "__main__":
    # TODO: You can run the code using the code below
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    # TODO: You should change this value accordingly
    # inflation_ratio = 3
    inflation_ratio = ROBOT_SIZE / resolution + 1
    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        planner.generate_plan()

    # You could replace this with other control publishers
    planner.publish_discrete_control()

    # save your action sequence
    result = np.array(planner.action_seq)
    if args.com:
        filename = "control_files/task1/1_com1_%d_%d.txt" % (goal[0], goal[1])
    else:
        filename = "control_files/task1/1_maze3_%d_%d.txt" % (goal[0], goal[1])
        # filename = "control_files/task2/2_maze0_%d_%d.txt" % (goal[0], goal[1])
    np.savetxt(filename, result, fmt="%.2e")

    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')

    # spin the ros
    rospy.spin()
