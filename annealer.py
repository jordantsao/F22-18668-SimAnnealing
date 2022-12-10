import random
from pprint import pprint
from math import log, exp
import time

"""
To obtain an initial average acceptance probability of around 80%, I chose an initial temperature T_0
such that
                  C_upper
            0.8 = ∫ exp(-Δc/T_0) dΔc
                  0
where C_upper is an arbitrary upper bound for cost difference

C_upper = 1.0 --> 2.15418
        = 0.5 --> 0.57051
        = 0.4 --> 0.31836
        = 0.3 --> 0.17434
        = 0.2 --> 0.08434
"""
TRIALS = 1000                   # number of trials to test final params
INITIAL_TEMPERATURE = 0.17434   # initial temperature T_0
N = 5                           # number of iterations between updating temperature
BETA = 1                        
ALPHA = 0.85                    # constant

# Various temperature schedules
LINEAR_SCHEDULE = 'Linear'
GEOMETRIC_SCHEDULE = 'Geometric'
EXPONENTIAL_SCHEDULE = 'Exponential'
SLOW_SCHEDULE = 'Slow'
LOGARITHMIC_SCHEDULE = 'Logarithmic'

ACCEPTANCE_ACCURACY = 0.77  # accuracy acceptance criterion
ACCEPTANCE_F1 = 0.55        # f1 acceptance criterion

class Annealer:
    
    def __init__(self, grid, model, dataset):
        self.data = dataset
        self.model = model
        self.grid = grid
        self.state = self.generate_initial_state()

        self.initial_temp = INITIAL_TEMPERATURE
        self.temp = self.initial_temp
        self.accuracy, self.f1 = self.train_and_test(self.state)[0:2]
        self.accuracy_opt, self.f1_opt = ACCEPTANCE_ACCURACY, ACCEPTANCE_F1

        self.beta = BETA
        self.alpha = ALPHA
        self.schedule = LOGARITHMIC_SCHEDULE

        self.iter = 0   # current iteration
        self.k = 0      # temperature cycle
        self.n = N      # temperature update interval

        self.print_simulation_status()
    
    # generate an initial state of parameters
    def generate_initial_state(self):
        state = {}
        for param, values in self.grid.items():             # for each parameter in the grid
            state[param] = random.randrange(0, len(values)) # select a random index of a parameter's gradients
        # pprint(state)
        return state
    
    # choose a state from the current state's neighborhood
    # this implementation distributes probability unevenly, favoring smaller gradients
    def generate_neighbor_state(self):
        neighbor = self.state.copy()
        param = random.choice(list(neighbor.keys()))                    # choose a random parameter
        neighbor[param] = random.randrange(0, len(self.grid[param]))    # choose a random value for the parameter
        return neighbor
    
    # reduce the temperature according to the temperature schedule
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

    # calculate the probability of changing to the neighboring state based on cost difference
    def calculate_energy_magnitude(self, cost):
        # if cost difference is negative --> change state
        # if cost difference is positive --> change state based on probability
        probability = 1.0 if cost <= 0 else exp((-cost) / self.temp)
        print(f'probability: {probability}')
        return probability
    
    # train the model with specified params 10 times and return average performance metrics
    def train_and_test(self, state):

        # set model parameters according to a given state
        params = {}
        for param, index in state.items():
            params[param] = self.grid[param][index]
        # print('PARAMS:')
        # pprint(params)
        self.model.set_params(params)

        # split dataset, train and test model x10
        accuracy, f1, precision, recall = 0, 0, 0, 0
        for i in range(10):
            self.data.split_train_test()
            self.model.train(self.data.X_train, self.data.Y_train)
            a, f, p, r = self.model.get_metrics(self.data.X_test, self.data.Y_test)
            accuracy += a
            f1 += f
            precision += p
            recall += r
        accuracy /= 10
        f1 /= 10
        precision /= 10
        recall /= 10
        return accuracy, f1, precision, recall
    
    # represent state as a string
    def get_state_string(self):
        state = ""
        for param, index in self.state.items():
            state += str(index)
        return state
    
    # print current simulation metrics
    def print_simulation_status(self):
        print("\nIteration " + str(self.iter) + " Cycle " + str(self.k))    # print iteration and temperature cycle
        print("Acc: " + str(self.accuracy) + " F1: " + str(self.f1))        # print state accuracy and f1 scores
        print("Current State: " + self.get_state_string())                  # print state as string

    # run simulation
    def begin_simulation(self):
        start = time.time()
        while ((self.accuracy < self.accuracy_opt or self.f1 < self.f1_opt) and self.temp > 1e-5):                     # stop condition
            neighbor = self.generate_neighbor_state()                       # select random neighboring state
            accuracy, f1, precision, recall = self.train_and_test(neighbor) # train and test on new params
            cost = (self.f1 - f1)                                           # calculate difference in cost
            probability = self.calculate_energy_magnitude(cost)             # calculate acceptance probability

            # accept new state according to calculated probability
            if random.uniform(0, 1) <= probability:
                self.state = neighbor
                self.accuracy = accuracy
                self.f1 = f1
            
            self.iter += 1                  # increment iteration
            if self.iter % self.n == 0:     # update temperature at specified interval
                self.reduce_temperature()
                self.k += 1
            self.print_simulation_status()  # print simulation metrics
        end = time.time()
        print(f'Duration: {end - start}')
    
    # get true average performance metrics for result state
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