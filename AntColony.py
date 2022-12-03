import random as rn
import numpy as np
from numpy.random import choice as np_choice
import time

# adapted from https://github.com/Akavall/AntColonyOptimization

class AntColony(object):

    def __init__(self, distances, n_ants=10, n_best=10, n_iterations=10, decay=0.95, p_weight=1, d_weight=1, time_limit=60):
        self.distances  = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_indexs = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = p_weight
        self.beta = d_weight
        self.time_limit = time_limit
        self.num_solutions = 0

    def run(self):
        shortest_path = None
        start_time = time.time()
        all_time_shortest_path = ("placeholder", np.inf)

        # loop for n_iterations
        for i in range(self.n_iterations):

            if time.time() - start_time > self.time_limit:
                break # time limit reached

            # get all the paths
            all_paths = self.gen_all_paths()

            # spread pheromone
            self.spread_pheronome(all_paths, self.n_best)

            # calculate shortest path
            if len(all_paths) == 0:
                continue # no valid paths
            shortest_path = min(all_paths, key=lambda x: x[1])
            # print (shortest_path)

            # update bssf if necessary
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path

            # decay pheromone
            self.pheromone = self.pheromone * self.decay

        return all_time_shortest_path, self.num_solutions

    def spread_pheronome(self, all_paths, n_best):
        # sort paths by distance
        sorted_paths = sorted(all_paths, key=lambda x: x[1])

        # for n_best number of paths
        for path, dist in sorted_paths[:n_best]:
            for step in path:
                # add pheromone to the step (more for shorter paths)
                self.pheromone[step] += 1.0 / self.distances[step]

    def gen_path_dist(self, path):
        total_dist = 0
        for step in path:
            total_dist += self.distances[step]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        # generate a path for each ant
        for i in range(self.n_ants):
            path = self.gen_path(0) # start at city 0
            if path == False:
                continue # ant could not find a valid path so don't add
            all_paths.append((path, self.gen_path_dist(path)))
            self.num_solutions += 1
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(self.pheromone[prev], self.distances[prev], visited)
            if move == False:
                return False # ant could not find a valid move
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start)) # going back to start city
        return path

    def pick_move(self, pheromone, dist, visited):
        try:
            # null chacne of visiting a city already visited
            pheromone = np.copy(pheromone)
            pheromone[list(visited)] = 0

            # calculate probabilities
            row = (pheromone ** self.alpha) * (( 1.0 / dist) ** self.beta)
            norm_row = row / row.sum()

            # pick move
            move = np_choice(self.all_indexs, 1, p=norm_row)[0]
            return move
        except:
            # no valid moves
            return False
