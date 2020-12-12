from MH import maximize
import pandas as pd
import numpy as np
from typing import Union
from scipy.spatial import ConvexHull, distance_matrix
import math, random
from MH import maximize
import pandas as pd
import numpy as np
from typing import Union
from scipy.spatial import ConvexHull, distance_matrix
import math, random

def compute_beta_vec(bo, epsilon_vec):
    """Compute beta from distribution of f."""
    sorted_f_vec = np.sort(bo.f_vec)
    f0 = sorted_f_vec[0]
    N0 = ((sorted_f_vec - f0) == 0).sum()
    f1_index = np.where((sorted_f_vec - f0) > 0)[0][0]
    f1 = sorted_f_vec[f1_index]
    N1 = ((sorted_f_vec - f1) == 0).sum()
    beta_vec = np.log(N1 / (N0 * epsilon_vec)) / (f1 - f0)
    return beta_vec

def get_optimal_betas_df(lambda_vec, epsilon_vec, lambda_arrays_list, model_string='G1'):
    """Return a dataframe containing the optimal values for beta."""
    running_list = []
    model_repeated = np.repeat(model_string, len(epsilon_vec))
    for i, lambda_array in enumerate(lambda_arrays_list):
        lambda_repeated = np.repeat(lambda_vec[i], len(epsilon_vec))
        l = list(zip(model_repeated, lambda_repeated, epsilon_vec, np.median(lambda_array, axis=0)))
        for item in l:
            running_list.append(item)

    optimal_betas_df = pd.DataFrame.from_dict(running_list).rename(columns={0: 'model', 1: 'lambda', 2: 'epsilon', 3: 'beta'})
    return optimal_betas_df

def getOptBetaSeq(generator_name: str, lambda_: Union[int, float, str]):
    beta_lookup_table = pd.read_csv('G1_and_G2_optimal_betas.csv')
    if generator_name in ('G1', 'G2'):
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == generator_name]['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        print(f'Original lambda is {round(lambda_, 2)} Corrected one is {round(correct_lambda, 2)}')
        
        betas = beta_lookup_table.query("(model == @generator_name) & (`lambda` == @correct_lambda)")
        return sorted(betas.beta)
    else:
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == 'G1']['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        betas1 = beta_lookup_table.query("(model == 'G1') & (`lambda` == @correct_lambda)")
        
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == 'G2']['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        betas2 = beta_lookup_table.query("(model == 'G2') & (`lambda` == @correct_lambda)")
        allbeta = np.array([sorted(betas1.beta), sorted(betas2.beta)])
        
        return np.mean(allbeta, axis=0)

def interpBetas(betas, steps, smooth=True):
    if steps <= len(betas):
        return betas
    # least square
    x = np.linspace(0, steps - 1, num=len(betas), endpoint=False)
    A = np.vstack([x, np.ones(len(x))]).T
    y = np.log(betas)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    smoothed = np.exp(coef[0] * np.arange(steps) + coef[1])
    if smooth:
        return smoothed
    # interpolation
    betas_new = betas.copy()
    betas_new.append(smoothed[-1])
    xp = np.linspace(0, steps - 1, num=len(betas_new))
    xp = np.int32(xp)
    beta_full =  np.interp(np.arange(steps), xp, betas_new)
    return beta_full

@maximize
def objective(v, x, lam, state):
    indicator = state
    n = len(indicator)
#     assert len(v) == n
#     assert x.shape == (n, 2)
    
    def diameter(array):
        hull = ConvexHull(array, incremental=False)
        edges = array[hull.vertices]
        diam = np.max(distance_matrix(edges, edges))
        return diam
    
    population = np.sum(v * indicator)
    cities = x[indicator]
    radius = diameter(cities) / 2
    area = math.pi * radius ** 2
    
    obj = population - lam * n * area
    return obj

def transition(state):
    ind = random.randint(0, len(state) - 1)
    new_state = np.copy(state)
    new_state[ind] = not state[ind]
    return new_state
def getOptBetaSeq(generator_name: str, lambda_: Union[int, float, str]):
    beta_lookup_table = pd.read_csv('G1_and_G2_optimal_betas.csv')
    if generator_name in ('G1', 'G2'):
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == generator_name]['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        print(f'Original lambda is {round(lambda_, 2)} Corrected one is {round(correct_lambda, 2)}')
        
        betas = beta_lookup_table.query("(model == @generator_name) & (`lambda` == @correct_lambda)")
        return sorted(betas.beta)
    else:
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == 'G1']['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        betas1 = beta_lookup_table.query("(model == 'G1') & (`lambda` == @correct_lambda)")
        
        aval_lambdas = pd.unique(beta_lookup_table[beta_lookup_table.model == 'G2']['lambda'])
        correct_lambda = aval_lambdas[np.argmin(np.abs(aval_lambdas - lambda_))]
        betas2 = beta_lookup_table.query("(model == 'G2') & (`lambda` == @correct_lambda)")
        allbeta = np.array([sorted(betas1.beta), sorted(betas2.beta)])
        
        return np.mean(allbeta, axis=0)

def interpBetas(betas, steps, smooth=True):
    if steps <= len(betas):
        return betas
    # least square
    x = np.linspace(0, steps - 1, num=len(betas), endpoint=False)
    A = np.vstack([x, np.ones(len(x))]).T
    y = np.log(betas)
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    smoothed = np.exp(coef[0] * np.arange(steps) + coef[1])
    if smooth:
        return smoothed
    # interpolation
    betas_new = betas.copy()
    betas_new.append(smoothed[-1])
    xp = np.linspace(0, steps - 1, num=len(betas_new))
    xp = np.int32(xp)
    beta_full =  np.interp(np.arange(steps), xp, betas_new)
    return beta_full

@maximize
def objective(v, x, lam, state):
    indicator = state
    n = len(indicator)
#     assert len(v) == n
#     assert x.shape == (n, 2)
    
    def diameter(array):
        hull = ConvexHull(array, incremental=False)
        edges = array[hull.vertices]
        diam = np.max(distance_matrix(edges, edges))
        return diam
    
    population = np.sum(v * indicator)
    cities = x[indicator]
    radius = diameter(cities) / 2
    area = math.pi * radius ** 2
    
    obj = population - lam * n * area
    return obj

def transition(state):
    ind = random.randint(0, len(state) - 1)
    new_state = np.copy(state)
    new_state[ind] = not state[ind]
    return new_state
