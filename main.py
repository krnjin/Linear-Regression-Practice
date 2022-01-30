from hashlib import new
from tkinter import N
from numpy import *
import sys

def run(input_file):

	#step 1 - collect our data
	try:
		points = genfromtxt(input_file, delimiter=',')
	except:
		print('Invalid input. Please enter the name of the input file you want to run linear regression on')
	else:
		#step 2 - define our hyperparam
		#how fast should our model converge(optimal model)
		# too small learning_rate = slow conversion
		# too big learning_rate = error function might not decrease
		learning_rate = 0.0001
		#y = mx + b (slope formula)
		initial_b = 0
		initial_m = 0
		num_iterations = 1000

		#step 3 - train our model
		print('starting gradient descent at b = {0}, m={1}, error = {2}'.format(initial_b, 
																				initial_m,
																				compute_error_for_line_given_points(initial_b, initial_m, points)))
		[b,m] = gradient_descent_runner(points, 
										initial_b,
										initial_m,
										learning_rate,num_iterations)
		print('running...\n\n')
		print('after {0} iterations, b = {1}, m={2}, error = {3}'.format(num_iterations,
																		b, 
																		m, 
																		compute_error_for_line_given_points(b,m,points)))

# computing the distance from the point and the predicted line
def compute_error_for_line_given_points(b, m, points):
	#initial error at 0
	total_error = 0

	#get all the value from the csv file
	for i in range(0, len(points)):

		#numpy vector - [a,b] where a is the row and b is the column
		x = points[i,0]
		y = points[i,1]
		#get the diff and square it and add it to the total
		total_error += (y - (m * x + b)) ** 2
	#get the average
	return total_error/float(len(points))

def gradient_descent_runner(points,starting_b,starting_m,learning_rates,num_iteration):
	#starting b and m value
	b = starting_b
	m = starting_m

	#perform gradient descent
	for i in range(num_iteration):
		#update b and m with the new more accurate b and m by performing this gradient step
		b, m = step_gradient(b, m, array(points),learning_rates)

	return [b, m]

####### 
def step_gradient(b_current, m_current, points, learning_rate):
	#initialize b and m gradient = starting point
	b_gradient = 0
	m_gradient = 0
	N = float(len(points))
	for i in range(0, len(points)):
		x = points[i,0]
		y = points[i,1]
		#direction with respect to b and m
		#compute partial derivatives of the error function
		b_gradient += (-2/N) * (y-((m_current*x)+b_current))
		m_gradient += (-2/N) * x * (y-((m_current*x)+b_current))

		#update b and m value
	new_b = b_current - (learning_rate * b_gradient)
	new_m = m_current - (learning_rate * m_gradient)
	return [new_b, new_m]

if __name__ == '__main__':
	run(sys.argv[1])
	