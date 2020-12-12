from math import exp
from random import random
import numpy as np


def minimize(func):
    def to_maximize(*args, **kwargs):
        value = func(*args, **kwargs)
        return -value
    return to_maximize


def maximize(func):
    def do_nothing(*args, **kwargs):
        value = func(*args, **kwargs)
        return value
    return do_nothing


class optimizer(object):
    """ 
    Metropolis-Hastings optimizer based on random walk (maximize objective function)
    """
    def __init__(self, obj_func, trans_func, beta, init_state, **kwargs):
        self.beta = beta
        self.obj_func = obj_func
        self.trans_func = trans_func
        self._kwargs = kwargs
        self.state = init_state
        self.obj_val = self.obj_func(state=init_state, **self._kwargs)

        self._init_state = init_state
    
    def reset(self):
        self.state = self._init_state
        self.obj_val = self.obj_func(state=self.state, **self._kwargs)
    
    def step(self):
        i = self.state
        j = self.trans_func(i)
        obj_j = self.obj_func(state=j, **self._kwargs)
        obj_i = self.obj_func(state=i, **self._kwargs)
        acceptance = self.accpet(obj_j, obj_i)
        if random() <= acceptance:
            self.state = j
            self.obj_val = obj_j
        return self.state, self.obj_val

    def accpet(self, j, i):
        rate = min(1, exp(self.beta * (j - i)))
        return rate
    
    def run(self, iters=100, beta_schedule=None, reset=True):
        if reset:
            self.reset()
        vals = []
        states = []
        for i in range(iters):
            if beta_schedule is not None:
                interval = iters // len(beta_schedule)
                beta_i = i // interval
                self.beta = beta_schedule[beta_i]
            curr_state, curr_val = self.step()
            vals.append(curr_val)
            states.append(curr_state)
        return vals, states
