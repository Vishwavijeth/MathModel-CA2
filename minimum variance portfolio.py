from scipy.optimize import minimize
import math
import numpy as np

data = {}

for ticker in data:
    data[ticker]['RoR'] = (data[ticker]['Close Price'] - data[ticker]['Open Price']) / data[ticker]['Open Price']

mu = np.array([data[ticker]['RoR'].mean() for ticker in data])
sigma = np.array([data[ticker]['RoR'].std() for ticker in data])

def weightConstraint(weights):
    return np.sum(weights) - 1.0

def minObjectiveFun(weights, covMatrix):
    return np.dot(weights, np.dot(covMatrix, weights))

def minVar(mu, cov):
    print("Minimizing Risk")
    initialWeights = np.ones(len(mu)) / len(mu)
    constraints = ({'type': 'eq', 'fun': weightConstraint})
    result = minimize(minObjectiveFun, initialWeights, args = (cov), method = 'SLSQP', constraints = constraints)
    optimalWeights = result.x

    print("Optimal Weights:", result.x)
    print("Portfolio Risk:", math.sqrt(result.fun))
    print('Portfolio Expected Return:', np.dot(optimalWeights, mu))

    return optimalWeights, math.sqrt(result.fun), np.dot(optimalWeights, mu)

covMatrix = np.array(np.cov([data[ticker]['RoR'] for ticker in data]))
mvp = minVar(mu, covMatrix)