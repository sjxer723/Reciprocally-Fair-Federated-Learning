import itertools
import math
import numpy as np
from random import randint


class FLInstance:
    def __init__(self, _n, _s, _alpha, _beta, _eps=0.1) -> None:
        self.n = _n  # the number of agents
        self.eps = _eps  # epsilon
        self.s = _s  # vector of s_i
        self.alpha = _alpha
        self.beta = _beta

    def update_s(self, s_dict):
        self.s = [s_dict[i] for i in range(self.n)]

    def compute_derivative_of_shapley_value(self, i, derivative_f):
        permutations = []
        num_perms = 0

        if i >= self.n:
            raise ValueError("Index out of range")
        if self.n <= 5:
            permutations = list(itertools.permutations(range(self.n)))
            num_perms = len(permutations)
        else:
            num_perms = int(self.n * np.log(self.n) / self.eps)
            permutations = set()
            for _ in range(num_perms):
                perm = np.random.permutation(self.n)
                permutations.add(tuple(perm))

        if num_perms == 0:
            raise ValueError("Empty permutation")

        shapley_values = []
        for perm in permutations:
            index_of_i = perm.index(i)
            agents_with_i = perm[: index_of_i + 1]  # agents that include i

            derivative = derivative_f(agents_with_i, i)
            shapley_values.append(derivative)
        return sum(shapley_values) / num_perms

    def compute_shapley_values(self, f):
        shapley_values = [0 for _ in range(self.n)]
        permutations = []

        if self.n <= 5:
            permutations = list(itertools.permutations(range(self.n)))
            num_perms = len(permutations)
        else:
            num_perms = int(self.n * np.log(self.n) / self.eps)
            permutations = set()
            for _ in range(num_perms):
                perm = np.random.permutation(self.n)
                permutations.add(tuple(perm))

        for perm in permutations:
            for i in range(self.n):
                agents_with_i = perm[: i + 1]
                agents_without_i = perm[:i]
                shapley_val_perm_i = f(agents_with_i) - f(agents_without_i)
                shapley_values[perm[i]] += shapley_val_perm_i
        shapley_values = [val / num_perms for val in shapley_values]

        return shapley_values

    # def compute_shapley_value_derivative(self, i):
    #     permutations = None
    #     shapley_values = []

    #     if i >= self.n:
    #         return 0
    #     if self.n <= 5:
    #         permutations = list(itertools.permutations(range(self.n)))
    #         num_perms = len(permutations)
    #     else:
    #         num_perms = int(self.n * np.log(self.n) / self.eps)
    #         permutations = set()
    #         for _ in range(num_perms):
    #             perm = np.random.permutation(self.n)
    #             permutations.add(tuple(perm))

    #     if len(permutations) == 0:
    #         return ValueError("Empty permutation")

    #     for perm in permutations:
    #         sum_of_s_before_i = self.s[i]
    #         for j in perm:
    #             if j == i:
    #                 break
    #             sum_of_s_before_i += self.s[j]
    #         if sum_of_s_before_i == 0:
    #             shapley_val = self.n
    #         else:
    #             ## the derivate is n * alpha * beta / (sum of s_j, j appears before i)^{beta +1}
    #             shapley_val = self.n * self.alpha * self.beta * 1.0 / pow(sum_of_s_before_i, self.beta+1)
    #             # pow(sum_of_s_before_i, self.beta + 1)
    #         shapley_values.append(shapley_val)
    #     return sum(shapley_values)/num_perms

    # def compute_shapley_value(self, i):
    #     permutations = None
    #     shapley_values = []

    #     if i >= self.n:
    #         return 0
    #     if self.n <= 5:
    #         permutations = list(itertools.permutations(range(self.n)))
    #         num_perms = len(permutations)
    #     else:
    #         num_perms = int(self.n * np.log(self.n) / self.eps)
    #         permutations = set()
    #         for _ in range(num_perms):
    #             perm = np.random.permutation(self.n)
    #             permutations.add(tuple(perm))

    #     if len(permutations) == 0:
    #         return ValueError("Empty permutation")

    #     for perm in permutations:
    #         sum_of_s_before_i = self.s[i]
    #         for j in perm:
    #             if j == i:
    #                 break
    #             sum_of_s_before_i += self.s[j]
    #         if sum_of_s_before_i == 0:
    #             shapley_val = 0
    #         elif sum_of_s_before_i == self.s[i]:
    #             shapley_val = self.n * (1- self.alpha / pow(sum_of_s_before_i, self.beta))
    #         else:
    #             shapley_val = self.n * (self.alpha / pow(sum_of_s_before_i - self.s[i], self.beta) - \
    #                                     self.alpha / pow(sum_of_s_before_i, self.beta))

    #         shapley_values.append(shapley_val)
    #     return sum(shapley_values)/num_perms
