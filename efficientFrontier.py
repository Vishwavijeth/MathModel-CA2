import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

dateparse = lambda x: datetime.strptime(x, '%B-%Y')
data = {}

for ticker in data:
    data[ticker]['RoR'] = (data[ticker]['Close Price'] - data[ticker]['Open Price']) / data[ticker]['Open Price']

mu = np.array([data[ticker]['RoR'].mean() for ticker in data])
sigma = np.array([data[ticker]['RoR'].std() for ticker in data])

print("Returns:", mu)
print("Risk:", sigma)

def portfolioPerformance(weights, mu, cov):
    returns = np.dot(weights, mu)
    std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    return returns, std

def generatePortfolios(n, mu, cov, rf):
  results = np.zeros((3, n))

  for i in range(n):
    weights = np.random.random(len(mu))
    weights /= np.sum(weights)
    mu_, sigma_ = portfolioPerformance(weights, mu, cov)
    results[0,i] = mu_
    results[1,i] = sigma_
    results[2,i] = (mu_ - rf) / sigma_  # Sharpe Ratio

  return results

covMatrix = np.array(np.cov([data[ticker]['RoR'] for ticker in data]))
results = generatePortfolios(1000, mu, covMatrix, 0.06)

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(results[0,:], results[1,:], c=results[2,:], marker='o')
plt.xlabel('Portfolio Standard Deviation (Risk)')
plt.ylabel('Portfolio Return')
plt.title('Efficient Frontier')
plt.colorbar(label='Sharpe Ratio')
plt.show()
