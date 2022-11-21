#!/usr/bin/python3
import math
from queue import PriorityQueue
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
elif PYQT_VER == 'PYQT4':
    from PyQt4.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools
from State import *

class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
		This is the entry point for the default solver
		which just finds a valid random tour.  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of solution, 
		time spent to find solution, number of permutations tried during search, the 
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
		This is the entry point for the greedy solver, which you must implement for 
		the group project (but it is probably a good idea to just do it for the branch-and
		bound project as a way to get your feet wet).  Note this could be used to find your
		initial BSSF.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found, the best
		solution found, and three null values for fields not used for this 
		algorithm</returns> 
	'''

    def greedy(self, time_allowance=60.0):
        pass

    ''' <summary>
		This is the entry point for the branch-and-bound algorithm that you will implement
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number solutions found during search (does
		not include the initial BSSF), the best solution found, and three more ints: 
		max queue size, total number of states created, and number of pruned states.</returns> 
	'''
    def branchAndBound(self, time_allowance=60.0):

        # init start variables
        num_solutions = 0
        random_tour = self.defaultRandomTour()
        bssf = State(np.array([[0]]), bound=random_tour['cost']) # initial bssf bound is just a random path
        max_queue_len = 0
        total_states = 0
        pruned_num = 0
        start_time = time.time()

        # queue
        states = PriorityQueue()
        states.put(self.build_start_matrix())

        # loop while states left in queue
        while states.qsize() > 0 and time.time() - start_time < time_allowance:
            total_states += 1
            max_queue_len = states.qsize() if states.qsize() > max_queue_len else max_queue_len

            current = states.get()
            # current = states.pop(-1)
            if current.bound >= bssf.bound:
                pruned_num += 1  # prune
                continue

            # generate children of current state
            children = current.expand()
            for child in children:
                if child.check_if_solution() and child.bound < math.inf:
                    num_solutions += 1
                    if child.bound < bssf.bound:
                        bssf = child
                elif child.bound < bssf.bound:
                    # states.append(child)
                    states.put(child)
                else:
                    pruned_num += 1 # prune child

        # get the route from the best found option
        route = self.get_route(bssf)

        # if no path was found better than the random tour return that
        if len(bssf.matrix) == 1:
            return random_tour

        # return results
        results = {}
        solution = TSPSolution(route)
        end_time = time.time()
        results['cost'] = solution.cost if num_solutions > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = num_solutions
        results['soln'] = solution
        results['max'] = max_queue_len
        results['total'] = total_states + states.qsize() # num popped + num left in queue
        results['pruned'] = pruned_num
        return results

    def build_start_matrix(self):
        # construct a matrix reprisenting the routes and the cities
        cities = self._scenario.getCities()
        matrix = np.empty(shape=(len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                matrix[i][j] = cities[i].costTo(cities[j])
        return State(matrix, 'Start matrix')

    def get_route(self, state):
        cities = self._scenario.getCities()
        route = []
        for i in state.route:
            route.append(cities[i])
        return route



    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''
    def fancy(self, time_allowance=60.0):
        pass
