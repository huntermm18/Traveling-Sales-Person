import math
import numpy as np

RUN_TESTS = False
PRINT_STATES = False
# RUN_TESTS = True
# PRINT_STATES = True

class State:
    state_num = 2  # for testing labels

    def __init__(self, matrix, name='', bound=0, a=0, b=None, parent=None, route=[0]):
        self.matrix = matrix.copy()
        self.name = name
        self.parent = parent
        self.bound = bound
        self.a = a
        self.b = b
        self.calc_reduced_cost_matrix(self.matrix, a, b)
        self.route = route
        self.cost = None
        if b:
            self.route.append(b)
        if PRINT_STATES:
            self.to_string()

    def __lt__(self, other):
        # prioritize route length and break tie breakers with smallest bound
        score1 = (len(self.route) * -1024) + (self.bound // 100)
        score2 = (len(other.route) * -1024) + (other.bound // 100)
        # lower score has higher priority
        return score1 < score2

    # calculate the state's reduced cost matrix
    def calc_reduced_cost_matrix(self, matrix, a, b):

        # make row and col inf for a and b
        if a is not None and b is not None:
            if matrix[a][b] == math.inf:
                self.bound = math.inf
                return  # a can't go to b so give up on this one
            matrix[a, :] = math.inf
            matrix[:, b] = math.inf
            matrix[b][a] = math.inf

        # iterate over rows
        for row in matrix:
            row_min = min(row)
            if row_min == math.inf:
                continue  # skip row
            for i in range(len(row)):
                row[i] -= row_min
            self.bound += row_min  # add min of row to bound

        # iterate over columns
        for i in range(len(matrix[0])):
            col = matrix[:, i]
            col_min = min(col)
            if col_min == math.inf:
                continue  # skip col
            for j in range(len(col)):
                col[j] -= col_min
            self.bound += col_min  # add min of col to bound

    # get all the children of the state
    def expand(self):
        if self.b == None:
            self.b = 0
        children = []
        for i in range(len(self.matrix[0])):
            if self.b == i:
                continue  # skip going to itself
            if (self.matrix[self.b, i]) == math.inf:
                continue  # skip
            child = State(self.matrix, str(self.state_num), self.bound+(self.matrix[self.b][i]), a=self.b, b=i, parent=self.name, route=self.route.copy())
            self.state_num += 1
            children.append(child)
        return children

    # see if this state could be a solution
    def check_if_solution(self):
        # if route contains all the cities
        if len(self.route) == len(self.matrix):
            return True
        return False

    # to string for testing
    def to_string(self):
        print(f'\nBound: {self.bound},  a: {self.a},  b: {self.b},  solution: {self.check_if_solution()}')
        print(f'route: {self.route}')
        if self.name != '':
            print(f'State {self.name},  Parent: State {self.parent}')
        s = [[str(e) for e in row] for row in self.matrix]
        lens = [max(map(len, col)) for col in zip(*s)]
        fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
        table = [fmt.format(*row) for row in s]
        print('\n'.join(table))


# for testing
test3 = np.array([
    [math.inf, 385.00000, 1801.00000, 371.00000],
    [math.inf, math.inf, 1693.00000, 639.00000],
    [2080.00000, 1533.00000, math.inf, 2131.00000],
    [373.00000, math.inf, 1855.00000, math.inf]
])

test_matrix = np.array([
    [math.inf, 7, 3, 12],
    [3, math.inf, 6, 14],
    [5, 8, math.inf, 6],
    [9, 3, 5, math.inf]
])
test_matrix2 = np.array([
    [math.inf, math.inf, math.inf, math.inf],
    [0.0, math.inf, math.inf, 10.0],
    [math.inf, 3.0, math.inf, 0.0],
    [6.0, 0.0, math.inf, math.inf]
])

testt = np.array([
    [math.inf, 605.0, 1941.0],
    [math.inf, 605.0, 1613.0],
    [1941.0, 1613.0, math.inf]
])


def run_state_tests(matrix):
    # initial
    state = State(matrix, 1)
    state.to_string()

    # children
    print('\nChildren...')
    children = state.expand()
    for c in children:
        print('---------------------')
        c.to_string()
        print('\nChildren...')
        for cc in c.expand():
            cc.to_string()
        print('---------------------')


if RUN_TESTS:
    run_state_tests(testt)
