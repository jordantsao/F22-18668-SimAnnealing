import random
from pprint import pprint
from math import log, exp
import time

SVC_GRID = {
    'kernel': ['linear', 'rbf', 'sigmoid'],         # kernel type
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],      # regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1, 10, 100, 1000],  # kernel coefficient for 'rbf', 'poly', and 'sigmoid'
    'degree': [1, 2, 3, 4],                         # for polynomial kernel
    'tol': [0.001, 0.01, 0.1, 1],                   # for stopping criterion
    'max_iter': [10, 50, 100, 500, 1000, -1]        # hard limit on iterations
}

"""
To obtain an initial average acceptance probability of around 80%, I chose an initial temperature T_0
such that
                  c
            0.8 = ∫ exp(-x/T_0) dx
                  0
"""
TRIALS = 100
# 2.15418 0.0855
INITIAL_TEMPERATURE = 0.31836
BETA = 1
ALPHA = 0.85
LINEAR_SCHEDULE = 'Linear'
GEOMETRIC_SCHEDULE = 'Geometric'
EXPONENTIAL_SCHEDULE = 'Exponential'
SLOW_SCHEDULE = 'Slow'
LOGARITHMIC_SCHEDULE = 'Logarithmic'

# Number of iterations between updating temperature
N = 5

class Simulation:
    
    def __init__(self, grid, model, dataset):
        self.data = dataset
        self.model = model
        self.grid = grid
        self.state = self.generate_initial_state()

        self.initial_temp = INITIAL_TEMPERATURE
        self.temp = self.initial_temp
        self.accuracy, self.f1 = self.train_and_test(self.state)[0:2]

        self.beta = BETA
        self.alpha = ALPHA
        self.schedule = LOGARITHMIC_SCHEDULE

        # current iteration
        self.iter = 0
        # temperature cycle
        self.k = 0
        # temperature update interval
        self.n = N

        self.print_simulation_status()
    
    def generate_initial_state(self):
        state = {}
        for param, values in self.grid.items():
            state[param] = random.randrange(0, len(values))
        # pprint(state)
        return state
    
    def generate_neighbor_state(self):
        neighbor = self.state.copy()
        param = random.choice(list(neighbor.keys()))
        neighbor[param] = random.randrange(0, len(self.grid[param]))
        return neighbor
    
    def reduce_temperature(self):
        print(f'Old temp: {self.temp}')
        if self.schedule == 'Linear':
            self.temp -= self.alpha
        elif self.schedule == 'Geometric':
            self.temp *= self.alpha
        elif self.schedule == 'Exponential':
            self.temp = self.initial_temp * exp(-(self.k**(1/18144)))
        elif self.schedule == 'Slow':
            self.temp = self.temp / (1 + self.beta * self.temp)
        elif self.schedule == 'Logarithmic':
            self.temp = self.initial_temp / (1 + log(1 + self.k))
        print(f'New temp: {self.temp}')

    def calculate_energy_magnitude(self, cost):
        probability = 1.0 if cost <= 0 else exp((-cost) / self.temp)
        print(f'probability: {probability}')
        return probability
    
    def train_and_test(self, state):
        params = {}
        for param, index in state.items():
            params[param] = self.grid[param][index]
        # print('PARAMS:')
        # pprint(params)
        self.model.set_params(params)
        self.data.split_train_test()
        self.model.train(self.data.X_train, self.data.Y_train)
        accuracy, f1, precision, recall = self.model.get_metrics(self.data.X_test, self.data.Y_test)
        return accuracy, f1, precision, recall
    
    def get_state_string(self):
        state = ""
        for param, index in self.state.items():
            state += str(index)
        return state
    
    def print_simulation_status(self):
        print("\nIteration " + str(self.iter) + " Cycle " + str(self.k))
        print("Acc: " + str(self.accuracy) + " F1: " + str(self.f1))
        print("Current State: " + self.get_state_string())

    def begin(self):
        start = time.time()
        while (self.accuracy < 0.85 or self.f1 < 0.7):
            neighbor = self.generate_neighbor_state()
            accuracy, f1, precision, recall = self.train_and_test(neighbor)
            cost = (self.f1 - f1) + (self.accuracy - accuracy)
            probability = self.calculate_energy_magnitude(cost)

            if random.uniform(0, 1) <= probability:
                self.state = neighbor
                self.accuracy = accuracy
                self.f1 = f1
            
            # increment iteration
            self.iter += 1
            # update temperature at specified interval
            if self.iter % self.n == 0:
                self.reduce_temperature()
                self.k += 1
            
            self.print_simulation_status()
        end = time.time()
        print(f'Duration: {end - start}')
            
    def test(self):
        avg_accuracy, avg_f1, avg_precision, avg_recall = 0, 0, 0, 0
        for i in range(TRIALS):
            accuracy, f1, precision, recall = self.train_and_test(self.state)
            avg_accuracy += accuracy
            avg_f1 += f1
            avg_precision += precision
            avg_recall += recall
        avg_accuracy /= TRIALS
        avg_f1 /= TRIALS
        avg_precision /= TRIALS
        avg_recall /= TRIALS
        print(f'Final Accuracy: {avg_accuracy}, Final F1: {avg_f1}, Final Precision: {avg_precision}, Final Recall: {avg_recall}')