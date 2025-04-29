import math
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
from sklearn.linear_model import LinearRegression

# Step 1: Define the chosen assets
assets = ['AAPL', 'MSFT', 'META', 'VZ', 'NKE', 'GS', 'SPY']

# Step 2: Retrieve daily close data for the past 2 years
end_date = dt.datetime.today()
start_date = end_date - dt.timedelta(days=2*365)

price_data = yf.download(assets, 
                         start=start_date, 
                         end=end_date,
                         )['Adj Close']
returns = price_data.pct_change().dropna()  # Calculate daily returns and drop NaN values

# Step 3: Define a class for portfolio statistics and efficient frontier
class PortfolioOptimizer:
    def __init__(self, returns):
        """
        Initialize the optimizer with returns data.
        """
        self.returns = returns  
        self.mean_returns = returns.mean()  
        self.cov_matrix = returns.cov()  
        self.num_assets = len(self.mean_returns)  
    
    def portfolio_performance(self, weights):
        """
        Calculate expected return and volatility for a given portfolio weight.
        """
        portfolio_return = np.sum(weights * self.mean_returns) * 252  
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std_dev  
    
    def monte_carlo_simulation(self, num_portfolios=10000):
        """
        Run Monte Carlo simulation to generate random portfolios.
        """
        results = np.zeros((3, num_portfolios))
        weights_record = []

        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            weights_record.append(weights)
            
            portfolio_return, portfolio_std_dev = self.portfolio_performance(weights)
            sharpe_ratio = portfolio_return / portfolio_std_dev
            
            results[0, i] = portfolio_std_dev  # Volatility
            results[1, i] = portfolio_return   # Return
            results[2, i] = sharpe_ratio       # Sharpe Ratio
            
        return results, weights_record  

    def max_sharpe_ratio(self):
        """
        Optimize the portfolio for maximum Sharpe ratio.
        """
        def negative_sharpe_ratio(weights):
            portfolio_return, portfolio_std_dev = self.portfolio_performance(weights)
            return -portfolio_return / portfolio_std_dev
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array(self.num_assets * [1. / self.num_assets,]) 
        
        optimal_result = sco.minimize(negative_sharpe_ratio, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal_result  

    def min_variance(self):
        """
        Optimize the portfolio for minimum variance.
        """
        def portfolio_volatility(weights):
            return self.portfolio_performance(weights)[1]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(self.num_assets))
        initial_guess = np.array(self.num_assets * [1. / self.num_assets,])
        
        optimal_result = sco.minimize(portfolio_volatility, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        return optimal_result 

    def display_efficient_frontier(self, results, weights_record):
        """
        Plot the efficient frontier with the Monte Carlo simulation results and fit a linear regression line.
        """
        # Plot Monte Carlo results
        plt.figure(figsize=(10, 7))
        plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis', marker='o', alpha=0.6)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Efficient Frontier with Linear Regression')
        
        # Mark the portfolio with the maximum Sharpe ratio
        max_sharpe = self.max_sharpe_ratio()
        max_sharpe_return, max_sharpe_volatility = self.portfolio_performance(max_sharpe.x)
        plt.scatter(max_sharpe_volatility, max_sharpe_return, color='r', marker='*', s=100, label='Max Sharpe Ratio')
        
        # Mark the portfolio with the minimum variance
        min_vol = self.min_variance()
        min_vol_return, min_vol_volatility = self.portfolio_performance(min_vol.x)
        plt.scatter(min_vol_volatility, min_vol_return, color='b', marker='*', s=100, label='Minimum Volatility')
        
        plt.legend()
        plt.savefig("efficient_frontier.png")  
        plt.show()

    def display_correlation_matrix(self):
        """
        Display the correlation matrix heatmap of asset returns.
        """
        sns.heatmap(self.returns.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Asset Returns')
        plt.savefig("correlation_matrix.png") 
        plt.show()

# Step 4: Run the Monte Carlo simulation and output optimal weights
optimizer = PortfolioOptimizer(returns)
sim_results, sim_weights = optimizer.monte_carlo_simulation()

# Step 5: Plot the efficient frontier with linear regression line
optimizer.display_efficient_frontier(sim_results, sim_weights)

# Step 6: Display the correlation matrix
optimizer.display_correlation_matrix()

# Step 7: Output the optimal weights for maximum Sharpe ratio
optimal_sharpe = optimizer.max_sharpe_ratio()
print("Optimal Weights for Maximum Sharpe Ratio:")
for asset, weight in zip(assets, optimal_sharpe.x):
    print(f"{asset}: {weight:.2%}")
