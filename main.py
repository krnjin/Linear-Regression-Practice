from numpy import *


def run():

	#step 1 - collect our data
	points = genfromtxt('data.csv',delimiter=',')

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
	print('starting gradient descent at b = {0}, m={1}, error = {2}'.format(initial_b, initial_m,compute_error_for_line_given_points(initial_b, initial_m,points)))
	[b,m] = gradient_descent_runner(points, initial_b,initial_m,learning_rate,num_iterations)

	print('ending point at at b = {1}, m={2}, error = {3}'.format(num_iterations,b, m, compute_error_for_line_given_points(b,m,points)))



if __name__ == '__main':
	run()


def compute_error_for_line_given_points(b, m, p):
	#initial error at 0
	total_error = 0

	for i in range(0, len(p)):
		x = 



def gradient_descent_runner(p,b,m,l,n):
	return