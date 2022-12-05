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
from AntColony import AntColony


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
        start_time = time.time()
        matrix = self.build_start_matrix().matrix
        route = []
        current_city = 0
        cost = 0
        cities = self._scenario.getCities()
        found = False
        closest_city = 0

        # matrix[:, 0] = math.inf

        while not found:
            # find closest city
            closest_city = 0
            for i in range(len(matrix)):
                if matrix[current_city][i] < matrix[current_city][closest_city]:
                    closest_city = i

            # solution found
            if len(route) == len(matrix):
                found = True  # succesful route
                continue

            # if you cant get to the city
            if matrix[current_city][closest_city] == math.inf:
                print('not found')
                return False

            # clear column and add to route
            cost += cities[current_city].costTo(cities[closest_city])
            matrix[:, closest_city] = math.inf
            route.append(closest_city)
            current_city = closest_city

        # reults
        route_f = []
        for i in range(len(route)):
            route_f.append(cities[route[i]])
        bssf = TSPSolution(route_f)

        # route.append(0)
        route_formatted = self.get_route(route)
        results = {}
        solution = TSPSolution(route_formatted)
        # results['soln'] = solution
        results['soln'] = bssf

        results['cost'] = bssf.cost
        results['time'] = time.time() - start_time
        results['count'] = None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

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
        max_queue_len = 0
        total_states = 0
        pruned_num = 0
        start_time = time.time()

        # start out with greedy solution
        start_tour = self.greedy()
        bssf = State(np.array([[0]]), bound=start_tour['cost'])  # initial bssf with greedy cost results

        # if greedy didn't work, take a sample of random tours and keep the best one
        if bssf.bound == math.inf:
            start_tour = self.defaultRandomTour()
            for i in range(5):
                temp = self.defaultRandomTour()
                if temp['cost'] < start_tour['cost']:
                    start_tour = temp
            bssf = State(np.array([[0]]), bound=start_tour['cost'])  # initial bssf with best random tour cost

        # queue
        states = PriorityQueue()
        states.put(self.build_start_matrix())

        # loop while states left in queue
        while states.qsize() > 0 and time.time() - start_time < time_allowance:
            max_queue_len = states.qsize() if states.qsize() > max_queue_len else max_queue_len

            current = states.get()
            if current.bound >= bssf.bound:
                pruned_num += 1  # prune
                continue

            # generate children of current state
            children = current.expand()
            for child in children:
                total_states += 1
                # check if child is a solution
                if child.check_if_solution() and child.bound < math.inf:
                    if child.bound < bssf.bound:
                        num_solutions += 1
                        bssf = child
                    else:
                        pruned_num += 1
                elif child.bound < bssf.bound:
                    states.put(child)
                else:
                    pruned_num += 1  # prune child

        # get the route from the best found option
        route = self.get_route(bssf.route)

        # if no path was found better than the initial bssf return that
        if len(bssf.matrix) == 1:
            print('using start tour')
            results = start_tour
            end_time = time.time()
            results['time'] = end_time - start_time
            results['count'] = num_solutions
            results['soln'] = start_tour['soln']
            results['max'] = max_queue_len
            results['total'] = total_states
            results['pruned'] = pruned_num
            return results

        # return results
        results = {}
        solution = TSPSolution(route)
        end_time = time.time()
        results['cost'] = solution.cost if num_solutions > 0 else math.inf
        results['time'] = end_time - start_time
        results['count'] = num_solutions
        results['soln'] = solution
        results['max'] = max_queue_len
        results['total'] = total_states
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

    def get_route(self, route):
        cities = self._scenario.getCities()
        result = []
        for i in route:
            result.append(cities[i])
        return result

    ''' <summary>
		This is the entry point for the algorithm you'll write for your group project.
		</summary>
		<returns>results dictionary for GUI that contains three ints: cost of best solution, 
		time spent to find best solution, total number of solutions found during search, the 
		best solution found.  You may use the other three field however you like.
		algorithm</returns> 
	'''

    def fancy(self, time_allowance=60.0):
        start_time = time.time()

        # create distance matrix
        cities = self._scenario.getCities()
        distances = np.empty(shape=(len(cities), len(cities)))
        for i in range(len(cities)):
            for j in range(len(cities)):
                distances[i][j] = cities[i].costTo(cities[j]) if cities[i].costTo(cities[j]) != 0 else 3.0

        try:
            # create the ant colony
            ant_colony = AntColony(distances, n_ants=150, n_best=110, n_iterations=100, decay=.95, p_weight=1, d_weight=1, time_limit=time_allowance)
            shortest_path, num_solutions = ant_colony.run()
            city_order = [cities[i[0]] for i in shortest_path[0]]
            solution = TSPSolution(city_order)
            # print("shorted_path: {}".format(shortest_path))

            # return results
            results = {}
            end_time = time.time()
            results['cost'] = solution.cost
            results['time'] = end_time - start_time
            results['count'] = num_solutions
            results['soln'] = solution
            results['max'] = None
            results['total'] = None
            results['pruned'] = None
            return results

        except:
            # no valid route found or other error occurred
            print('\nFancy failed. Using random. Try increasing n_ants or n_iterations.\n')
            return self.defaultRandomTour()

