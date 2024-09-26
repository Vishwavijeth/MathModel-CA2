import numpy as np
import math
from scipy.optimize import minimize

data = {}

for ticker in data:
    data[ticker]['RoR'] = (data[ticker]['Close Price'] - data[ticker]['Open Price']) / data[ticker]['Open Price']

mu = np.array([data[ticker]['RoR'].mean() for ticker in data])
sigma = np.array([data[ticker]['RoR'].std() for ticker in data])

covMatrix = np.array(np.cov([data[ticker]['RoR'] for ticker in data]))

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

mvp = minVar(mu, covMatrix)

def covariance(x, y):
    n = len(x)
    meanX = np.mean(x)
    meanY = np.mean(y)
    total = 0
    for i in range(n):
        total += (x[i] - meanX) * (y[i] - meanY)
    return total / n

beta = {}
betaV = 0

for i, ticker in enumerate(data):
    X = list(data[ticker]['RoR'])
    Y = list(data['ICICI']['RoR'])

    beta[ticker] = covariance(X, Y) / np.var(Y)
    betaV += mvp[0][i] * beta[ticker]

print("Beta Values:")
print(beta)
print("Beta Value of MVP:", betaV)
