from __future__ import division
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def linear_regression(x, y):
	n = len(x)

	# Initialize terms
	b0 = 0
	b1 = 0
	term_1 = 0
	term_2 = 0
	term_3 = 0
	term_4 = 0
	
	# Find terms used to determine b0 and b1
	for i in xrange(n):	
		term_1 += x[i]
		term_2 += y[i]
		term_3 += x[i]**2
		term_4 += x[i]*y[i]

	b0 = (term_3*term_2 - term_4*term_1) / (n*term_3 - (term_1)**2)
	b1 = (n*term_4 - (term_1*term_2)) / (n*term_3 - (term_1)**2)

	return b0, b1


def plot(X, Y, X_reg, Y_reg, b0, b1, mse):
	plt.ion()

	# Plots
	plt.plot(X, Y, 'ro')
	plt.plot(X_reg, Y_reg, 'b--')

	# Labels
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('Linear Regression\n y = {:.3f} + {:.3f}x\n MSE: {}'.format(b0, b1, mse))
	
	# Max mins for axis
	min_x, max_x = min(X)-1, max(X)+1
	min_y, max_y = min(Y)-2, max(Y)+2
	plt.axis([min_x, max_x, min_y, max_y])

	# Pause
	plt.pause(6.)


def get_mse(Y, Y_reg):
	mse = 0
	for y, y_reg in zip(Y, Y_reg):
		mse += np.abs(y - y_reg)

	return mse


def main(input_path):

	# Load data frame from csv file
	df = pd.read_csv(input_path, header=0, dtype=float)

	# Extract X and Y columns from dataframe
	X = df['x']
	Y = df['y']

	# Linear regression to find b0, b1
	b0, b1 = linear_regression(X, Y)

	# Setup the X, Y for linear regression
	X_reg = np.arange(min(X), max(X), 0.01)
	Y_reg = [(b0 + b1*x) for x in X_reg]

	# Calculate MSE
	MSE = get_mse(Y, Y_reg)
	
	# Plot
	plot(X, Y, X_reg, Y_reg, b0, b1, MSE)
	

if __name__ == '__main__':
	input_path = './data/input.csv'
	main(input_path)


