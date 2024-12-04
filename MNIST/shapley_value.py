import itertools
import math
import numpy as np
from random import randint

class FLInstance:
    def __init__(self, _n, _s, _alpha, _beta, _eps=0.001) -> None:
        self.n = _n         # the number of agents
        self.eps = _eps     # epsilon
        self.s = _s         # vector of s_i
        self.alpha = _alpha
        self.beta = _beta

    def update_s(self, s_dict):
        self.s = [s_dict[i] for i in range(self.n)]

    def compute_shapley_value_derivative(self, i):
        permutations = None
        shapley_values = []

        if i >= self.n:
            return 0
        if self.n <= 5:
            permutations = list(itertools.permutations(range(self.n)))
            num_perms = len(permutations)
        else:
            num_perms = int(self.n * np.log(self.n) / self.eps)
            permutations = set()
            for _ in range(num_perms):
                perm = np.random.permutation(self.n)
                permutations.add(tuple(perm))

        if len(permutations) == 0:
            return ValueError("Empty permutation")
        
        for perm in permutations:
            sum_of_s_before_i = self.s[i]
            for j in perm:
                if j == i:
                    break
                sum_of_s_before_i += self.s[j]
            if sum_of_s_before_i == 0:
                shapley_val = self.n
            else:
                ## the derivate is n * alpha * beta / (sum of s_j, j appears before i)^{beta +1}
                shapley_val = self.n * self.alpha * self.beta * 1.0 / pow(sum_of_s_before_i, self.beta+1) 
                # pow(sum_of_s_before_i, self.beta + 1)
            shapley_values.append(shapley_val)
        return sum(shapley_values)/num_perms
    
    def compute_shapley_value(self, i):
        permutations = None
        shapley_values = []

        if i >= self.n:
            return 0
        if self.n <= 5:
            permutations = list(itertools.permutations(range(self.n)))
            num_perms = len(permutations)
        else:
            num_perms = int(self.n * np.log(self.n) / self.eps)
            permutations = set()
            for _ in range(num_perms):
                perm = np.random.permutation(self.n)
                permutations.add(tuple(perm))

        if len(permutations) == 0:
            return ValueError("Empty permutation")
        
        for perm in permutations:
            sum_of_s_before_i = self.s[i]
            for j in perm:
                if j == i:
                    break
                sum_of_s_before_i += self.s[j]
            if sum_of_s_before_i == 0:
                shapley_val = 0
            elif sum_of_s_before_i == self.s[i]:
                shapley_val = self.n * (1- self.alpha / pow(sum_of_s_before_i, self.beta))
            else:
                shapley_val = self.n * (self.alpha / pow(sum_of_s_before_i - self.s[i], self.beta) - \
                                        self.alpha / pow(sum_of_s_before_i, self.beta))

            shapley_values.append(shapley_val)
        return sum(shapley_values)/num_perms