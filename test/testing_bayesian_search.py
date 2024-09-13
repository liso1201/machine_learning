import sys
sys.path.append('.')

import numpy as np
import matplotlib.pyplot as plt

from search.methods import bayesian_search
from tune import defaultScores

from tests.random_dataframe import generate_randomData_for_tests

#MODELO DE TEST
from sklearn.ensemble import GradientBoostingRegressor

train, test = generate_randomData_for_tests(n_vars=3, lenght=1000, test_size=0.2)

xtrain = train.iloc[:, :-1]
ytrain = train.iloc[:, -1]

xtest = test.iloc[:, :-1]
ytest = test.iloc[:, -1]

preditor = GradientBoostingRegressor()

params = dict(
        loss = ['squared_error', 'absolute_error'],
        criterion = ['friedman_mse', 'squared_error'],
        learning_rate = np.linspace(0.0, 0.9, 5, dtype=float),
        n_estimators = np.linspace(100, 5000, 5, dtype=int),
        max_depth = np.linspace(100, 5000, 5, dtype=int),
        min_impurity_decrease = np.linspace(0.01, 0.95, 5, dtype=float),
        min_samples_split = np.linspace(2, 100, 5, dtype=int),
        min_samples_leaf = np.linspace(2, 100, 5, dtype=int)
    )


bestModel, search = bayesian_search(
    params, 
    preditor, 
    xtrain, ytrain, xtest, ytest, 
    n_iters=200, 
    scores=defaultScores, 
    n_samples=5)

metric = 'mean_squared_error'

fig = plt.figure(figsize=(2*3.55, 2*3.55))

ncols = 3
nrows = len(params) // ncols + 1 or 1

for i, p in enumerate(params, start=1):
    ax = fig.add_subplot(nrows, ncols, i)

    dfi = search[[p, metric]].sort_values(by=p, ignore_index=True)

    ax.plot(p, metric, data=dfi, color='black', zorder=0)
    
    minPoint = dfi.get(metric).argmin()
    ax.scatter(dfi.loc[minPoint, p], dfi.loc[minPoint, metric], color='red', s=35, zorder=1, label='Global minimum')

    ax.set_xlabel(p, fontsize='small')
    ax.set_ylabel(metric, fontsize='small')
    ax.tick_params('both', labelsize='small')

fig.tight_layout()
plt.show()
