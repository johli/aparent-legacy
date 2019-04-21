
"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import os
import sys
import time
import subprocess
import shlex
import os.path

import numpy
import math

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
#from theano.tensor.signal import pool


from theano.tensor.nnet import conv

import theano.sparse as Tsparse

from logistic_sgd_global_onesided2 import LogisticRegression, load_input_data, load_output_data
from mlp import HiddenLayer

#import pylab as pl
#import matplotlib.cm as cm
import matplotlib as mpl
mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.sparse as sp
import scipy.io as spio

import weblogolib

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

class ModifiedBackprop(object):

	def __init__(self, nonlinearity):
		self.nonlinearity = nonlinearity
		self.ops = {}  # memoizes an OpFromGraph instance per tensor type

	def __call__(self, x):
		# OpFromGraph is oblique to Theano optimizations, so we need to move
		# things to GPU ourselves if needed.
		'''if theano.sandbox.cuda.cuda_enabled:
			maybe_to_gpu = theano.sandbox.cuda.as_cuda_ndarray_variable
		else:
			maybe_to_gpu = lambda x: x'''
		# We move the input to GPU if needed.
		#x = maybe_to_gpu(x)
		# We note the tensor type of the input variable to the nonlinearity
		# (mainly dimensionality and dtype); we need to create a fitting Op.
		tensor_type = x.type
		# If we did not create a suitable Op yet, this is the time to do so.
		if tensor_type not in self.ops:
			# For the graph, we create an input variable of the correct type:
			inp = tensor_type()
			# We pass it through the nonlinearity (and move to GPU if needed).
			outp = self.nonlinearity(inp)#maybe_to_gpu(self.nonlinearity(inp))
			# Then we fix the forward expression...
			op = theano.OpFromGraph([inp], [outp])
			# ...and replace the gradient with our own (defined in a subclass).
			op.grad = self.grad
			# Finally, we memoize the new Op
			self.ops[tensor_type] = op
		# And apply the memoized Op to the input we got.
		return self.ops[tensor_type](x)

class OrthogonalInit(object) :

	def __init__(self, gain=1.0):
		if gain == 'relu':
			gain = numpy.sqrt(2)
		self.gain = gain

	def sample(self, shape, rng):
		if len(shape) < 2:
			raise RuntimeError("Only shapes of length 2 or more are supported.")

		flat_shape = (shape[0], numpy.prod(shape[1:]))
		a = rng.normal(0.0, 1.0, flat_shape)
		u, _, v = numpy.linalg.svd(a, full_matrices=False)
		# pick the one with the correct shape
		q = u if u.shape == flat_shape else v
		q = q.reshape(shape)

		print('Doing Orthog. Init.')

		return numpy.asarray(self.gain * q, dtype=theano.config.floatX)


class GuidedBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		dtype = inp.dtype
		return (grd * (inp > 0).astype(dtype) * (grd > 0).astype(dtype),)

class ZeilerBackprop(ModifiedBackprop):
	def grad(self, inputs, out_grads):
		(inp,) = inputs
		(grd,) = out_grads
		#return (grd * (grd > 0).astype(inp.dtype),)  # explicitly rectify
		return (self.nonlinearity(grd),)

class LeNetConvPoolLayer(object):
	"""Pool Layer of a convolutional network """

	def store_w(self, w_file, W):
		numpy.save(w_file, W)
	
	def store_b(self, b_file, b):
		numpy.save(b_file, b)
	
	def store_model(self, W, b):
		self.store_w(self.store_as_w_file, W)
		self.store_b(self.store_as_b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)
	
	def __init__(self, orthoInit, rng, input, deactivated_filter, deactivated_output, filter_shape, image_shape, poolsize=(2, 2), stride=(1, 1), activation_fn=T.tanh, load_model = False, filter_init_pwms=None, filter_knockdown=None, filter_knockdown_init='rand', w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
		"""
		Allocate a LeNetConvPoolLayer with shared variable internal parameters.

		:type rng: numpy.random.RandomState
		:param rng: a random number generator used to initialize weights

		:type input: theano.tensor.dtensor4
		:param input: symbolic image tensor, of shape image_shape

		:type filter_shape: tuple or list of length 4
		:param filter_shape: (number of filters, num input feature maps,
							  filter height, filter width)

		:type image_shape: tuple or list of length 4
		:param image_shape: (batch size, num input feature maps,
							 image height, image width)

		:type poolsize: tuple or list of length 2
		:param poolsize: the downsampling (pooling) factor (#rows, #cols)
		"""

		self.w_file = w_file
		self.b_file = b_file
		self.store_as_w_file = w_file
		self.store_as_b_file = b_file
		if store_as_w_file is not None and store_as_b_file is not None :
			self.store_as_w_file = store_as_w_file
			self.store_as_b_file = store_as_b_file
		
		assert image_shape[1] == filter_shape[1]

		# there are "num input feature maps * filter height * filter width"
		# inputs to each hidden unit
		fan_in = numpy.prod(filter_shape[1:])
		# each unit in the lower layer receives a gradient from:
		# "num output feature maps * filter height * filter width" /
		#   pooling size
		fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
				   numpy.prod(poolsize))
		# initialize weights with random weights
		W_bound = numpy.sqrt(6. / (fan_in + fan_out))
		
		if load_model == False :
			W_values = numpy.asarray(
				rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
				dtype=theano.config.floatX
			)
			if orthoInit is not None :
				W_values = orthoInit.sample(filter_shape, rng)

			if filter_init_pwms is not None :
				#Squash PWM into fans (and scale by 2)
				filter_init_pwms = ((filter_init_pwms * 2.0 * W_bound) - W_bound) * 2

				W_values[0:filter_init_pwms.shape[0], 0, :, :] = filter_init_pwms[:, :, :]

			self.W = theano.shared(value=W_values, name='W', borrow=True)

			# the bias is a 1D tensor -- one bias per output feature map
			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)
		else :
			W_values = self.load_w(w_file + '.npy')
			b_values = self.load_b(b_file + '.npy')
			self.W = theano.shared(value=W_values, name='W', borrow=True)
			self.b = theano.shared(value=b_values, name='b', borrow=True)

			if filter_knockdown is not None :
				if filter_knockdown_init == 'rand' :
					W_values[filter_knockdown, :, :, :] = numpy.asarray(rng.uniform(low=-W_bound, high=W_bound, size=W_values[filter_knockdown, :, :, :].shape) / 2., dtype=theano.config.floatX)
					b_values[filter_knockdown] = numpy.zeros((len(filter_knockdown),), dtype=theano.config.floatX)
				elif filter_knockdown_init in ['zeros', 'hardzeros'] :
					W_values[filter_knockdown, :, :, :] = numpy.zeros(W_values[filter_knockdown, :, :, :].shape, dtype=theano.config.floatX)
					b_values[filter_knockdown] = numpy.zeros((len(filter_knockdown),), dtype=theano.config.floatX)

			if filter_init_pwms is not None :
				#Squash PWM into fans (and scale by 2)
				filter_init_pwms = ((filter_init_pwms * 2.0 * W_bound) - W_bound) * 2

				W_values[0:filter_init_pwms.shape[0], 0, :, :] = filter_init_pwms[:, :, :]

			self.W = theano.shared(value=W_values, name='W', borrow=True)
			self.b = theano.shared(value=b_values, name='b', borrow=True)

		# convolve input feature maps with filters
		conv_out = conv.conv2d(
			input=input,
			filters=self.W,
			filter_shape=filter_shape,
			subsample=stride,
			image_shape=image_shape
		)

		'''if(use_relu == True):
			activation = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
		else:
			activation = T.tanh(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))'''
		activation = activation_fn(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

		# downsample each feature map individually, using maxpooling
		pooled_out = downsample.max_pool_2d(
			input=activation,
			ds=poolsize,
			ignore_border=True
		)
		'''pooled_out = pool.pool_2d(
			input=activation,
			ws=poolsize,
			ignore_border=True,
			mode='max'
		)'''

		self.conv_out = conv_out
		self.activation = activation

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		
		self.output = pooled_out

		if filter_knockdown is not None and filter_knockdown_init == 'hardzeros' :
			self.W = T.set_subtensor(self.W[filter_knockdown, :, :, :], 0)
		
		# store parameters of this layer
		self.params = [self.W, self.b]

def relu(x):
    return T.switch(x<0, 0, x)

class DualCNN(object):

	def set_saliency_functions(self, data_set):
		data_set_x, data_set_y, data_set_L, data_set_d = data_set

		index = T.lscalar()
		batch_size = self.batch_size
		
		self.n_batches = data_set_x.get_value(borrow=True).shape[0] / batch_size
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y
		L_input = self.L_input
		d_input = self.d_input
		train_drop = self.train_drop

		outp = self.output_layer.s_y_given_x
		max_outp = outp[:,1]#T.max(outp, axis=1)
		input_saliency_from_output = theano.grad(max_outp.sum(), wrt=x_left)

		self.compute_input_saliency_from_output = theano.function(
			[index],
			[input_saliency_from_output],
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)

		out_conv0 = self.layer0_left.activation
		out_conv1 = self.layer1.activation

		conv0_saliency_from_output = theano.grad(max_outp.sum(), wrt=out_conv0)
		conv1_saliency_from_output = theano.grad(max_outp.sum(), wrt=out_conv1)

		self.compute_conv0_saliency_from_output = theano.function(
			[index],
			[conv0_saliency_from_output],
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)

		self.compute_conv1_saliency_from_output = theano.function(
			[index],
			[conv1_saliency_from_output],
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)

		#(batch_size, nkerns[0], 88, 1)
		filter_index = T.lscalar()
		activation_index = T.lscalar()

		input_saliency_from_conv0 = theano.grad(out_conv0[0, filter_index, activation_index, 0], wrt=x_left)
		input_saliency_from_conv1 = theano.grad(out_conv1[0, filter_index, activation_index, 0], wrt=x_left)

		self.compute_input_saliency_from_conv0 = theano.function(
			[index, filter_index, activation_index],
			[input_saliency_from_conv0],
			givens={
				x_left: self.reshape_datapoint(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_datapoint(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1])
			}
		)

		self.compute_input_saliency_from_conv1 = theano.function(
			[index, filter_index, activation_index],
			[input_saliency_from_conv1],
			givens={
				x_left: self.reshape_datapoint(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_datapoint(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1])
			}
		)


	def get_input_conv0_saliency(self, i, k, j) :
		saliency = self.compute_input_saliency_from_conv0(i, k, j)
		return saliency[0]
	def get_input_conv1_saliency(self, i, k, j) :
		saliency = self.compute_input_saliency_from_conv1(i, k, j)
		return saliency[0]

	def get_conv0_saliency(self):
		saliency = numpy.concatenate([self.compute_conv0_saliency_from_output(i) for i in xrange(self.n_batches)], axis=0)
		return saliency

	def get_conv1_saliency(self):
		saliency = numpy.concatenate([self.compute_conv1_saliency_from_output(i) for i in xrange(self.n_batches)], axis=0)
		return saliency

	def get_saliency(self):
		saliency = numpy.concatenate([self.compute_input_saliency_from_output(i) for i in xrange(self.n_batches)], axis=0)
		return saliency

	def set_data(self, data_set_x, data_set_y, data_set_L, data_set_d):
		index = T.lscalar()
		batch_size = self.batch_size
		
		self.n_batches = data_set_x.get_value(borrow=True).shape[0] / batch_size
		
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y
		L_input = self.L_input
		d_input = self.d_input
		train_drop = self.train_drop

		deactivated_filter_level1 = self.deactivated_filter_level1
		deactivated_output_level1 = self.deactivated_output_level1
		
		self.compute_logloss = theano.function(
			[index],
			self.output_layer.log_loss(y),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: data_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.compute_rsquare = theano.function(
			[index],
			self.output_layer.rsquare(y),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: data_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.compute_sse = theano.function(
			[index],
			self.output_layer.sse(y),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: data_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.compute_sst = theano.function(
			[index],
			self.output_layer.sst(y),
			givens={
				#x: data_set_x[index * batch_size: (index + 1) * batch_size],
				y: data_set_y[index * batch_size: (index + 1) * batch_size]
			}
		)
		self.compute_abs_error = theano.function(
			[index],
			self.output_layer.abs_error(y),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: data_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.predict = theano.function(
			[index],
			self.output_layer.recall(),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.class_score = theano.function(
			[index],
			self.output_layer.s_y_given_x,
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		
		data_x = T.tensor3('x')
		data_L = T.matrix('L_i')
		data_d = T.matrix('d_i')
		self.online_predict = theano.function(
			[data_x, data_L, data_d],
			self.output_layer.recall(),
			givens={
				x_left: data_x[:,randomized_regions[0][0]:randomized_regions[0][1]],
				x_right: data_x[:,randomized_regions[1][0]:randomized_regions[1][1]],
				L_input: data_L[:, :],
				d_input: data_d[:, :]
				,train_drop: 0
			}
		)

	def get_prediction(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.predict(i) for i in xrange(self.n_batches)])
		else:
			return self.predict(i)

	def get_class_score(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.class_score(i) for i in xrange(self.n_batches)])
		else:
			return self.class_score(i)

	def get_online_prediction(self, data_x):
		return self.online_predict(data_x)
	
	def get_rsquare(self):
		sses = [self.compute_sse(i) for i in xrange(self.n_batches)]
		ssts = [self.compute_sst(i) for i in xrange(self.n_batches)]
		return 1.0 - (numpy.sum(sses) / numpy.sum(ssts))
	
	def get_mean_abs_error(self):
		abs_errors = [self.compute_abs_error(i) for i in xrange(self.n_batches)]
		return numpy.mean(abs_errors)
	
	def get_rmse(self):
		sses = [self.compute_sse(i) for i in xrange(self.n_batches)]
		return numpy.sqrt(numpy.sum(sses) / (self.n_batches * self.batch_size))
	
	def get_logloss(self):
		losses = [self.compute_logloss(i) for i in xrange(self.n_batches)]
		return numpy.sum(losses) / (self.n_batches * self.batch_size)
	
	def reshape_batch(self, data_set_x, index, left_input_bound, right_input_bound):
		batch_size = self.batch_size
		num_features = self.num_features
		left_random_size = self.left_random_size
		right_random_size = self.right_random_size
		input_size = self.input_size
		
		reshaped_batch = Tsparse.basic.dense_from_sparse(data_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, input_size, num_features))[:,left_input_bound:right_input_bound]
		if batch_size == 1:
			reshaped_batch = T.unbroadcast(reshaped_batch, 0)
		return reshaped_batch.astype(theano.config.floatX)

	def reshape_datapoint(self, data_set_x, index, left_input_bound, right_input_bound):
		batch_size = self.batch_size
		num_features = self.num_features
		left_random_size = self.left_random_size
		right_random_size = self.right_random_size
		input_size = self.input_size
		
		reshaped_batch = Tsparse.basic.dense_from_sparse(data_set_x[index:index+1, :]).reshape((batch_size, input_size, num_features))[:,left_input_bound:right_input_bound]
		if batch_size == 1:
			reshaped_batch = T.unbroadcast(reshaped_batch, 0)
		return reshaped_batch.astype(theano.config.floatX)


	def generate_radial_sequence_logos(self, test_set):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer0_left = self.layer0_left

		index = T.lscalar()
		batch_size = self.batch_size
		
		input_x = test_set_x.eval()

		L_index = numpy.ravel(numpy.argmax(test_set_L.eval(), axis=1))

		n_batches = input_x.shape[0] / batch_size

		n_original = input_x.shape[0]
		
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y
		L_input = self.L_input

		num_filters = 70#50#30#70#128#256#128#70

		activation_length = 185 - 7

		filter_index = T.lscalar()
		get_layer0_activations_k = theano.function(
			[index, filter_index],
			layer0_left.activation[:, filter_index, :, :],
			givens={
				x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
			},
			on_unused_input='ignore'
		)



		#activations = numpy.concatenate([get_layer0_activations(i) for i in xrange(n_batches)], axis=0)

		print('Computed layer activations')

		
		
		#input_x = numpy.asarray(input_x.todense()).reshape((activations.shape[0], self.input_size, self.num_features))[:, 0:self.left_random_size, :]

		y_test_hat = numpy.array(self.get_prediction())
		y_test = numpy.array(test_set_y.eval()[:y_test_hat.shape[0],1])

		print('mean(y_test_hat) = ' + str(numpy.mean(y_test_hat)))
		print('mean(y_test) = ' + str(numpy.mean(y_test)))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
		logodds_test = safe_log(y_test / (1 - y_test))

		logodds_test_isinf = numpy.isinf(logodds_test)
		y_test_hat = y_test_hat[logodds_test_isinf == False]
		y_test = y_test[logodds_test_isinf == False]
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		logodds_test = logodds_test[logodds_test_isinf == False]
		#activations = activations[logodds_test_isinf == False, :, :, :]
		input_x = input_x[logodds_test_isinf == False, :]
		L_index = L_index[logodds_test_isinf == False]

		print('Reshaped and filtered activations')

		valid_testset_size = n_original - len(numpy.nonzero(logodds_test_isinf)[0])

		#logodds_test_hat injection
		logodds_test = logodds_test_hat
		y_test = y_test_hat

		logodds_test_avg = numpy.average(logodds_test)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

		pos_r = numpy.zeros((num_filters, activation_length))

		filter_width = 8
		
		

		libvar_2 = ('X' * (4)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('X' * 20) + ('X' * (88 - 7))
		libvar_2_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_2[i] == 'V' :
				libvar_2_id[i] = 1

		libvar_8 = ('X' * (4)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('V' * (20 - 7)) + ('X' * (88 - 7 + 7))
		libvar_8_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_8[i] == 'V' :
				libvar_8_id[i] = 1

		libvar_5 = ('X' * (4)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('X' * 20) + ('X' * (88 - 7))
		libvar_5_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_5[i] == 'V' :
				libvar_5_id[i] = 1

		libvar_11 = ('X' * (4)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('V' * (20 - 7)) + ('X' * (88 - 7 + 7))
		libvar_11_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_11[i] == 'V' :
				libvar_11_id[i] = 1

		#APA_SYM_PRX
		libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
		libvar_20_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_20[i] == 'V' :
				libvar_20_id[i] = 1

		'''libvar_21 = ('X' * (15 - 1)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (85 - 7 + 7))
		libvar_21_id = numpy.zeros(255 - 7)
		for i in range(0, 255 - 7) :
			if libvar_21[i] == 'V' :
				libvar_21_id[i] = 1'''

		#APA_SIMPLE
		libvar_22 = ('X' * 4) + ('V' * (147 - 7)) + ('X' * (34 + 7 - 7))
		libvar_22_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_22[i] == 'V' :
				libvar_22_id[i] = 1

		#APA_SIX
		libvar_30 = ('X' * (2 + 12)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 12))
		libvar_30_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_30[i] == 'V' :
				libvar_30_id[i] = 1

		libvar_31 = ('X' * (2 + 7)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 7))
		libvar_31_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_31[i] == 'V' :
				libvar_31_id[i] = 1

		libvar_32 = ('X' * (2 + 12)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 12))
		libvar_32_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_32[i] == 'V' :
				libvar_32_id[i] = 1

		libvar_33 = ('X' * (2 + 0)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 0))
		libvar_33_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_33[i] == 'V' :
				libvar_33_id[i] = 1

		libvar_34 = ('X' * (2 + 15)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 15))
		libvar_34_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_34[i] == 'V' :
				libvar_34_id[i] = 1

		libvar_35 = ('X' * (2 + 11)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 11))
		libvar_35_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_35[i] == 'V' :
				libvar_35_id[i] = 1


		pos_to_libs = [
			#[libvar_2_id, 2],
			#[libvar_8_id, 8],
			#[libvar_5_id, 5],
			#[libvar_11_id, 11],
			[libvar_20_id, 20],
			#[libvar_21_id, 21],
			#[libvar_22_id, 22],
			#[libvar_30_id, 30],
			#[libvar_31_id, 31],
			#[libvar_32_id, 32],
			#[libvar_33_id, 33],
			#[libvar_34_id, 34],
			#[libvar_35_id, 35],
		]
		pos_to_libs_lookup = []
		for pos in range(0, len(pos_to_libs[0][0])) :
			valid_libs = []
			valid_libs_str = ''
			for libvar in pos_to_libs :
				if libvar[0][pos] == 1 :
					valid_libs.append(libvar[1])
					valid_libs_str += '_' + str(libvar[1])
			pos_to_libs_lookup.append([pos, valid_libs, valid_libs_str])


		k_r = numpy.zeros(num_filters)

		f = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/avg_filter_r.txt', 'w')

		meme = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/tomtom_meme.txt', 'w')
		meme.write('MEME version 4\n')
		meme.write('\n')
		meme.write('ALPHABET= ACGT')
		meme.write('\n')
		meme.write('strands: + -')
		meme.write('\n')


		mer8 = []
		bases =['A', 'C', 'G', 'T']
		for b1 in bases :
			for b2 in bases :
				for b3 in bases :
					for b4 in bases :
						for b5 in bases :
							for b6 in bases :
								for b7 in bases :
									for b8 in bases :
										mer8.append(b1 + b2 + b3 + b4 + b5+ b6 + b7 + b8)


		#(num_data_points, num_filters, seq_length, 1)
		for k in range(0, num_filters) :


			filter_activations = numpy.concatenate([get_layer0_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((n_original, activation_length))
			#print(filter_activations.shape)
			filter_activations = filter_activations[logodds_test_isinf == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			#valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)

			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

			pos_activation = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			if len(spike_index) <= 0 :
				continue

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :]
			filter_L = L_index[spike_index]

			filter_logodds_test = logodds_test[spike_index]
			filter_y_test = y_test[spike_index]


			kernel = numpy.fliplr(numpy.flipud(numpy.array(layer0_left.W.eval())[k, 0, :, :]))
			bases = 'ACGT'
			
			eight_mers = []
			
			motif_weights = numpy.zeros(4**8)
			
			for i1 in range(0,4):
				for i2 in range(0,4):
					for i3 in range(0,4):
						for i4 in range(0,4):
							for i5 in range(0,4):
								for i6 in range(0,4):
									for i7 in range(0,4):
										for i8 in range(0,4):
											motif = bases[i1] + bases[i2] + bases[i3] + bases[i4] + bases[i5] + bases[i6] + bases[i7] + bases[i8]
											eight_mers.append(motif)
											motif_weights[i1 * 4**7 + i2 * 4**6 + i3 * 4**5 + i4 * 4**4 + i5 * 4**3 + i6 * 4**2 + i7 * 4 + i8] = kernel[0, i1] + kernel[1, i2] + kernel[2, i3] + kernel[3, i4] + kernel[4, i5] + kernel[5, i6] + kernel[6, i7] + kernel[7, i8]

			eight_mers = numpy.array(eight_mers)

			highest_motif_weight_index = numpy.argsort(motif_weights)[::-1]
			motif_weights = motif_weights[highest_motif_weight_index]
			eight_mers = eight_mers[highest_motif_weight_index]

			top_n = 5
			motif_center_blacklist_dict = {}
			for top_curr in range(0, top_n) :
				motif_center = ''
				motif_center_index = -1
				for kk in range(0, len(eight_mers)) :
					if eight_mers[kk] not in motif_center_blacklist_dict :
						motif_center_blacklist_dict[eight_mers[kk]] = True
						motif_center = eight_mers[kk]
						motif_center_index = kk
						break
				if motif_center == '' :
					break

				if motif_weights[kk] <= 0 :
					break

				motif_whitelist_dict = {}
				motif_whitelist_dict[motif_center] = True

				#Add all 1-nt neighbors
				for pos1 in range(0, 8) :
					for b1 in ['A', 'C', 'G', 'T'] :
						motif_neighbor = motif_center[:pos1] + b1 + motif_center[pos1+1:]

						motif_whitelist_dict[motif_neighbor] = True
						motif_center_blacklist_dict[motif_neighbor] = True

				#Add all 2-nt neighbors
				for pos1 in range(0, 8) :
					for pos2 in range(pos1 + 1, 8) :
						for b1 in ['A', 'C', 'G', 'T'] :
							for b2 in ['A', 'C', 'G', 'T'] :
								motif_neighbor = motif_center[:pos1] + b1 + motif_center[pos1+1:pos2] + b2 + motif_center[pos2+1:]

								#motif_whitelist_dict[motif_neighbor] = True
								motif_center_blacklist_dict[motif_neighbor] = True

				#Run pipeline on strict subset of motif candidates in whitelist


				max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

				max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

				top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))[::-1]
				#top_scoring_index = top_scoring_index[len(top_scoring_index)-3000:]#5000

				top_selection_limit = 3000
				top_selection_curr = 0
				
				PFM = numpy.zeros((filter_width, self.num_features))
				for ii in range(0, len(top_scoring_index)) :
					i = top_scoring_index[ii]

					filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
					filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :]
					#filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :] * filter_activations[i, max_spike[i]]
					#filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :] * filter_y_test[i]

					motif_candidate = translate_matrix_to_seq(filter_input)
					if motif_candidate not in motif_whitelist_dict :
						continue

					if top_selection_curr >= top_selection_limit :
						break
					top_selection_curr += 1

					PFM = PFM + filter_input * filter_activations[i, max_spike[i]]

				if top_selection_curr <= 200 :
					continue

				print('Motif ' + str(k) + '_' + str(top_curr))


				#Estimate PPM motif properties
				PPM = numpy.zeros(PFM.shape)
				for i in range(0, PFM.shape[0]) :
					if numpy.sum(PFM[i, :]) > 0 :
						PPM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])

				meme.write('MOTIF Layer_1_Filter_' + str(k) + '_' + str(top_curr) + '\n')
				meme.write('letter-probability matrix: alength= 4 w= 8 nsites= ' + str(top_selection_curr) + '\n')
				for i in range(0, PPM.shape[0]) :
					for j in range(0, 4) :
						meme.write(' ' + str(round(PPM[i, j], 6)) + ' ')
					meme.write('\n')
				meme.write('\n')





				#Calculate Pearson r
				'''logodds_test_curr = logodds_test
				logodds_test_avg_curr = logodds_test_avg
				logodds_test_std_curr = logodds_test_std

				max_activation_regions = [
					[numpy.ravel(max_activation[k, :]), 'All region'],
					[numpy.ravel(max_activation_up[k, :]), 'Upstream'],
					[numpy.ravel(max_activation_pas[k, :]), 'PAS'],
					[numpy.ravel(max_activation_dn[k, :]), 'Downstream']
				]
				r_up_k = 0
				r_pas_k = 0
				r_dn_k = 0
				for region in max_activation_regions :
					max_activation_k = region[0]

					max_activation_k = max_activation_k[L_index > 5]
					logodds_test_curr = logodds_test[L_index > 5]

					max_activation_k_avg = numpy.average(max_activation_k)
					max_activation_k_std = numpy.sqrt(numpy.dot(max_activation_k - max_activation_k_avg, max_activation_k - max_activation_k_avg))

					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					cov = numpy.dot(logodds_test_curr - logodds_test_avg_curr, max_activation_k - max_activation_k_avg)
					r = cov / (max_activation_k_std * logodds_test_std_curr)
					print(region[1] + ' r = ' + str(round(r, 2)))
					f.write('Filter ' + str(k) + ', ' + region[1] + ' r = ' + str(round(r, 2)) + '\n')

				print('')

				prev_selection_libs_str = 'X'
				for pos in range(0, activation_length) :

					pos_activation_curr = pos_activation
					logodds_test_curr = logodds_test
					logodds_test_avg_curr = logodds_test_avg
					logodds_test_std_curr = logodds_test_std
					curr_selection_libs_str = ''
					if pos_to_libs_lookup[pos][2] == '' :
						continue

					pos_to_lib = pos_to_libs_lookup[pos]
					curr_selection_libs_str = pos_to_lib[2]
					if curr_selection_libs_str == prev_selection_libs_str :
						pos_activation_curr = pos_activation_prev
						logodds_test_curr = logodds_test_prev
					else :
						whitelist_index = []
						for i in range(0, len(L_index)) :
							if L_index[i] in pos_to_lib[1] :
								whitelist_index.append(i)
						
						pos_activation_curr = pos_activation[whitelist_index, :]
						logodds_test_curr = logodds_test[whitelist_index]
					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					if curr_selection_libs_str == '' :
						continue

					pos_activation_k_pos = numpy.ravel(pos_activation_curr[:, pos])
					pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
					pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

					cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k_pos - pos_activation_k_pos_avg)
					r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_std_curr)

					if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
						pos_r[k, pos] = r_k_pos

					prev_selection_libs_str = curr_selection_libs_str
					pos_activation_prev = pos_activation_curr
					logodds_test_prev = logodds_test_curr'''


				logo_name = "avg_motif_" + str(k) + "_" + str(top_curr) + ".svg"
				logo_name_prob = "avg_motif_" + str(k) + "_" + str(top_curr) + "_prob.svg"

				logotitle = "Layer 1 Filter " + str(k) + '_' + str(top_curr)
				self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + logo_name, 8, logotitle=logotitle)#r
				#self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + logo_name_prob, 8, logotitle=logotitle, u='probability')#r

		f.close()
		meme.close()


		#All-filter positional Pearson r
		'''f = plt.figure(figsize=(32, 16))

		avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))
		pos_r = pos_r[avg_r_sort_index, :]

		plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Prox. selection Pearson r for all filters')
		#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
		#xticks = mer_sorted
		plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 185], [0 - 49, 25 - 49, 50 - 49, 75 - 49, 100 - 49, 125 - 49, 150 - 49, 175 - 49, 185 - 49])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

		plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "r_pos.svg")
		plt.close()'''


	def generate_sequence_logos_level2(self, test_set):
			test_set_x, test_set_y, test_set_L, test_set_d = test_set
			self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)
			self.set_saliency_functions((test_set_x, test_set_y, test_set_L, test_set_d))

			layer1 = self.layer1

			index = T.lscalar()
			batch_size = self.batch_size
			
			input_x = test_set_x.eval()

			L_index = numpy.ravel(numpy.argmax(test_set_L.eval(), axis=1))

			n_batches = input_x.shape[0] / batch_size
			
			randomized_regions = self.randomized_regions
			
			x_left = self.x_left
			x_right = self.x_right
			y = self.y
			L_input = self.L_input

			num_filters = 110#90#110#90#110#512#256#110

			activation_length = 84

			input_index = numpy.arange(input_x.shape[0])

			filter_index = T.lscalar()
			get_layer1_activations_k = theano.function(
				[index, filter_index],
				layer1.activation[:, filter_index, :, :],
				givens={
					x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
				},
				on_unused_input='ignore'
			)

			print('Computing layer activations')
			
			y_test_hat = numpy.array(self.get_prediction())
			y_test = numpy.array(test_set_y.eval()[:y_test_hat.shape[0],1])

			input_x = input_x[:y_test_hat.shape[0],:]
			L_index = L_index[:y_test_hat.shape[0]]
			input_index = input_index[:y_test_hat.shape[0]]

			logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
			logodds_test = safe_log(y_test / (1 - y_test))

			logodds_test_isinf = numpy.isinf(logodds_test)
			logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
			logodds_test = logodds_test[logodds_test_isinf == False]
			input_x = input_x[logodds_test_isinf == False, :]
			L_index = L_index[logodds_test_isinf == False]
			input_index = input_index[logodds_test_isinf == False]

			print('Reshaped and filtered activations')

			valid_testset_size = y_test_hat.shape[0] - len(numpy.nonzero(logodds_test_isinf)[0])

			#logodds_test_hat injection
			logodds_test = logodds_test_hat

			logodds_test_avg = numpy.average(logodds_test)
			logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

			max_activation = numpy.zeros((num_filters, valid_testset_size))
			max_activation_up = numpy.zeros((num_filters, valid_testset_size))
			max_activation_pas = numpy.zeros((num_filters, valid_testset_size))
			max_activation_dn = numpy.zeros((num_filters, valid_testset_size))

			pos_r = numpy.zeros((num_filters, activation_length))

			filter_width = 19
		
			#No-padding Library variation strings

			#APA_SYM_PRX
			libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
			libvar_20_id = numpy.zeros(185 - 7)
			for i in range(0, 185 - 7) :
				if libvar_20[i] == 'V' :
					libvar_20_id[i] = 1

			libvar_20_idd = numpy.zeros(84)
			for i in range(0, 84) :
				if libvar_20_id[2 * i] == 1 and libvar_20_id[2 * i + 1] == 1 :
					libvar_20_idd[i] = 1
			libvar_20_id = libvar_20_idd

			#APA_SIMPLE
			libvar_22 = ('X' * 4) + ('V' * (147 - 7)) + ('X' * (34 + 7 - 7))
			libvar_22_id = numpy.zeros(185 - 7)
			for i in range(0, 185 - 7) :
				if libvar_22[i] == 'V' :
					libvar_22_id[i] = 1

			libvar_22_idd = numpy.zeros(84)
			for i in range(0, 84) :
				if libvar_22_id[2 * i] == 1 and libvar_22_id[2 * i + 1] == 1 :
					libvar_22_idd[i] = 1
			libvar_22_id = libvar_22_idd


			pos_to_libs = [
				#[libvar_2_id, 2],
				#[libvar_8_id, 8],
				#[libvar_5_id, 5],
				#[libvar_11_id, 11],
				#[libvar_20_id, 20],
				[libvar_22_id, 22],
			]
			pos_to_libs_lookup = []
			for pos in range(0, len(pos_to_libs[0][0])) :
				valid_libs = []
				valid_libs_str = ''
				for libvar in pos_to_libs :
					if libvar[0][pos] == 1 :
						valid_libs.append(libvar[1])
						valid_libs_str += '_' + str(libvar[1])
				pos_to_libs_lookup.append([pos, valid_libs, valid_libs_str])

			k_r = numpy.zeros(num_filters)

			f = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/avg_filter_r.txt', 'w')

			f_cons = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/avg_filter_highseqs.txt', 'w')
			meme = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/tomtom_meme.txt', 'w')
			meme.write('MEME version 4\n')
			meme.write('\n')
			meme.write('ALPHABET= ACGT')
			meme.write('\n')
			meme.write('strands: + -')
			meme.write('\n')

			for k in range(0, num_filters) :

				filter_activations = numpy.concatenate([get_layer1_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((y_test_hat.shape[0], activation_length))
				filter_activations = filter_activations[logodds_test_isinf == False, :]

				valid_activations = numpy.zeros(filter_activations.shape)
				#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
				valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
				
				filter_activations = numpy.multiply(filter_activations, valid_activations)


				total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

				max_activation[k, :] = numpy.ravel(numpy.max(filter_activations, axis=1))

				#Region-specific max activations
				max_activation_up[k, :] = numpy.ravel(numpy.max(filter_activations[:,:22], axis=1))
				max_activation_pas[k, :] = numpy.ravel(numpy.max(filter_activations[:,22:25], axis=1))
				max_activation_dn[k, :] = numpy.ravel(numpy.max(filter_activations[:,25:47], axis=1))

				pos_activation = filter_activations[:, :]
				
				spike_index = numpy.nonzero(total_activations > 0)[0]

				if len(spike_index) <= 0 :
					continue

				filter_activations = filter_activations[spike_index, :]

				filter_inputs = input_x[spike_index, :]
				filter_L = L_index[spike_index]
				filter_input_index = input_index[spike_index]

				max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

				top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
				top_scoring_index = top_scoring_index[len(top_scoring_index)-300:]#300

				logo_width = filter_width


				PFM = numpy.zeros((filter_width, self.num_features))
				for ii in range(0, len(top_scoring_index)) :
					i = top_scoring_index[ii]

					'''input_saliency_from_conv1 = self.get_input_conv1_saliency(filter_input_index[i], k, max_spike[i])
					input_saliency_from_conv1_index = input_saliency_from_conv1 > 0
					input_saliency_from_conv1_id = numpy.zeros(input_saliency_from_conv1.shape)
					input_saliency_from_conv1_id[input_saliency_from_conv1_index] = 1
					input_saliency_from_conv1_id = input_saliency_from_conv1_id[0, 2*max_spike[i]:2*max_spike[i]+filter_width, :]'''

					filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
					
					filter_input = filter_input[2*max_spike[i]:2*max_spike[i]+filter_width, :] * filter_activations[i, max_spike[i]]
					#filter_input = numpy.multiply(filter_input[2*max_spike[i]:2*max_spike[i]+filter_width, :], input_saliency_from_conv1_id) #* filter_activations[i, max_spike[i]]

					#filter_input = numpy.multiply(filter_inputs[i, 2*max_spike[i]:2*max_spike[i]+filter_width, :], input_saliency_from_conv1_id) #* filter_activations[i, max_spike[i]]
					#filter_input = filter_inputs[i, 2*max_spike[i]:2*max_spike[i]+filter_width, :]

					#filter_input = numpy.multiply(filter_input, input_saliency_from_conv1_id) #* filter_activations[i, max_spike[i]]

					f_cons.write('Filter_' + str(k) + '\tTop_' + str(ii) + '\tAct_' + str(round(filter_activations[i, max_spike[i]], 1)) + '\t' + translate_matrix_to_seq(filter_input) + '\n')

					PFM = PFM + filter_input

				
				#print(k)
				#print(PFM)
				print('(Layer 2) Motif ' + str(k))
                
				#Estimate PPM motif properties
				PPM = numpy.zeros(PFM.shape)
				for i in range(0, PFM.shape[0]) :
					if numpy.sum(PFM[i, :]) > 0 :
						PPM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])

				meme.write('MOTIF Layer_2_Filter_' + str(k) + '\n')
				meme.write('letter-probability matrix: alength= 4 w= 19 nsites= 300\n')
				for i in range(0, PPM.shape[0]) :
					for j in range(0, 4) :
						meme.write(' ' + str(round(PPM[i, j], 6)) + ' ')
					meme.write('\n')
				meme.write('\n')

				#Calculate Pearson r
				logodds_test_curr = logodds_test
				logodds_test_avg_curr = logodds_test_avg
				logodds_test_std_curr = logodds_test_std

				max_activation_regions = [
					[numpy.ravel(max_activation[k, :]), 'All region'],
					[numpy.ravel(max_activation_up[k, :]), 'Upstream'],
					[numpy.ravel(max_activation_pas[k, :]), 'PAS'],
					[numpy.ravel(max_activation_dn[k, :]), 'Downstream']
				]
				for region in max_activation_regions :
					max_activation_k = region[0]

					max_activation_k = max_activation_k[L_index > 5]
					logodds_test_curr = logodds_test[L_index > 5]

					max_activation_k_avg = numpy.average(max_activation_k)
					max_activation_k_std = numpy.sqrt(numpy.dot(max_activation_k - max_activation_k_avg, max_activation_k - max_activation_k_avg))

					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					cov = numpy.dot(logodds_test_curr - logodds_test_avg_curr, max_activation_k - max_activation_k_avg)
					r = cov / (max_activation_k_std * logodds_test_std_curr)
					print(region[1] + ' r = ' + str(round(r, 2)))
					f.write('Filter ' + str(k) + ', ' + region[1] + ' r = ' + str(round(r, 2)) + '\n')

				print('')

				prev_selection_libs_str = 'X'
				for pos in range(0, activation_length) :

					pos_activation_curr = pos_activation
					logodds_test_curr = logodds_test
					logodds_test_avg_curr = logodds_test_avg
					logodds_test_std_curr = logodds_test_std
					curr_selection_libs_str = ''
					if pos_to_libs_lookup[pos][2] == '' :
						continue

					pos_to_lib = pos_to_libs_lookup[pos]
					curr_selection_libs_str = pos_to_lib[2]
					if curr_selection_libs_str == prev_selection_libs_str :
						pos_activation_curr = pos_activation_prev
						logodds_test_curr = logodds_test_prev
					else :
						whitelist_index = []
						for i in range(0, len(L_index)) :
							if L_index[i] in pos_to_lib[1] :
								whitelist_index.append(i)
						
						pos_activation_curr = pos_activation[whitelist_index, :]
						logodds_test_curr = logodds_test[whitelist_index]
					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					if curr_selection_libs_str == '' :
						continue

					pos_activation_k_pos = numpy.ravel(pos_activation_curr[:, pos])
					pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
					pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

					cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k_pos - pos_activation_k_pos_avg)
					r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_std_curr)

					if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
						pos_r[k, pos] = r_k_pos

					prev_selection_libs_str = curr_selection_libs_str
					pos_activation_prev = pos_activation_curr
					logodds_test_prev = logodds_test_curr

				logo_name = "avg_motif_" + str(k) + ".svg"
				logo_name_normed = "avg_motif_" + str(k) + '_normed' + ".svg"

				logotitle = "Layer 2 Filter " + str(k)
				self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + logo_name, logo_width, logotitle=logotitle)
				#self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + logo_name_normed, 19, normalize=True, logotitle=logotitle)

			f.close()
			f_cons.close()
			meme.close()

			#All-filter positional Pearson r
			f = plt.figure(figsize=(18, 16))

			avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))
			pos_r = pos_r[avg_r_sort_index, :]

			plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
			plt.colorbar()

			plt.xlabel('Sequence position')
			plt.title('Prox. selection Pearson r for all layer 2 filters')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.xticks([0, 12, 24, 36, 48, 60, 72, 84], [0 - 24, 12 - 24, 24 - 24, 36 - 24, 48 - 24, 60 - 24, 72 - 24, 84 - 24])
			plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

			plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

			#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + "r_pos.png")
			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + "r_pos.svg")
			plt.close()

			f = plt.figure(figsize=(18, 16))

			plt.pcolor(numpy.repeat(pos_r, 2, axis=1),cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
			plt.colorbar()

			plt.xlabel('Sequence position')
			plt.title('Prox. selection Pearson r for all layer 2 filters')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.xticks([0*2, 12*2, 24*2, 36*2, 48*2, 60*2, 72*2, 84*2], [0*2 - 24*2, 12*2 - 24*2, 24*2 - 24*2, 36*2 - 24*2, 48*2 - 24*2, 60*2 - 24*2, 72*2 - 24*2, 84*2 - 24*2])
			plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

			plt.axis([0, pos_r.shape[1] * 2, 0, pos_r.shape[0]])

			#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + "r_pos_projected.png")
			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter_level2/' + "r_pos_projected.svg")
			plt.close()
			

	def generate_sequence_logos(self, test_set):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer0_left = self.layer0_left

		index = T.lscalar()
		batch_size = self.batch_size
		
		input_x = test_set_x.eval()

		L_index = numpy.ravel(numpy.argmax(test_set_L.eval(), axis=1))

		n_batches = input_x.shape[0] / batch_size

		n_original = input_x.shape[0]
		
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y
		L_input = self.L_input

		num_filters = 70#50#30#70#128#256#128#70

		activation_length = 185 - 7

		filter_index = T.lscalar()
		get_layer0_activations_k = theano.function(
			[index, filter_index],
			layer0_left.activation[:, filter_index, :, :],
			givens={
				x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
			},
			on_unused_input='ignore'
		)



		#activations = numpy.concatenate([get_layer0_activations(i) for i in xrange(n_batches)], axis=0)

		print('Computed layer activations')

		
		
		#input_x = numpy.asarray(input_x.todense()).reshape((activations.shape[0], self.input_size, self.num_features))[:, 0:self.left_random_size, :]

		y_test_hat = numpy.array(self.get_prediction())
		y_test = numpy.array(test_set_y.eval()[:y_test_hat.shape[0],1])

		print('mean(y_test_hat) = ' + str(numpy.mean(y_test_hat)))
		print('mean(y_test) = ' + str(numpy.mean(y_test)))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
		logodds_test = safe_log(y_test / (1 - y_test))

		logodds_test_isinf = numpy.isinf(logodds_test)
		y_test_hat = y_test_hat[logodds_test_isinf == False]
		y_test = y_test[logodds_test_isinf == False]
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		logodds_test = logodds_test[logodds_test_isinf == False]
		#activations = activations[logodds_test_isinf == False, :, :, :]
		input_x = input_x[logodds_test_isinf == False, :]
		L_index = L_index[logodds_test_isinf == False]

		print('Reshaped and filtered activations')

		valid_testset_size = n_original - len(numpy.nonzero(logodds_test_isinf)[0])

		#logodds_test_hat injection
		logodds_test = logodds_test_hat
		y_test = y_test_hat

		logodds_test_avg = numpy.average(logodds_test)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

		max_activation = numpy.zeros((num_filters, valid_testset_size))
		max_activation_up = numpy.zeros((num_filters, valid_testset_size))
		max_activation_pas = numpy.zeros((num_filters, valid_testset_size))
		max_activation_dn = numpy.zeros((num_filters, valid_testset_size))
		#pos_activation = numpy.zeros((num_filters, y_test_hat.shape[0], activation_length))

		pos_r = numpy.zeros((num_filters, activation_length))

		filter_width = 8
		
		

		libvar_2 = ('X' * (4)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('X' * 20) + ('X' * (88 - 7))
		libvar_2_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_2[i] == 'V' :
				libvar_2_id[i] = 1

		libvar_8 = ('X' * (4)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('V' * (20 - 7)) + ('X' * (88 - 7 + 7))
		libvar_8_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_8[i] == 'V' :
				libvar_8_id[i] = 1

		libvar_5 = ('X' * (4)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('X' * 20) + ('X' * (88 - 7))
		libvar_5_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_5[i] == 'V' :
				libvar_5_id[i] = 1

		libvar_11 = ('X' * (4)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('V' * (20 - 7)) + ('X' * (88 - 7 + 7))
		libvar_11_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_11[i] == 'V' :
				libvar_11_id[i] = 1

		#APA_SYM_PRX
		libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
		libvar_20_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_20[i] == 'V' :
				libvar_20_id[i] = 1

		'''libvar_21 = ('X' * (15 - 1)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (85 - 7 + 7))
		libvar_21_id = numpy.zeros(255 - 7)
		for i in range(0, 255 - 7) :
			if libvar_21[i] == 'V' :
				libvar_21_id[i] = 1'''

		#APA_SIMPLE
		libvar_22 = ('X' * 4) + ('V' * (147 - 7)) + ('X' * (34 + 7 - 7))
		libvar_22_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_22[i] == 'V' :
				libvar_22_id[i] = 1

		#APA_SIX
		libvar_30 = ('X' * (2 + 12)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 12))
		libvar_30_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_30[i] == 'V' :
				libvar_30_id[i] = 1

		libvar_31 = ('X' * (2 + 7)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 7))
		libvar_31_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_31[i] == 'V' :
				libvar_31_id[i] = 1

		libvar_32 = ('X' * (2 + 12)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 12))
		libvar_32_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_32[i] == 'V' :
				libvar_32_id[i] = 1

		libvar_33 = ('X' * (2 + 0)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 0))
		libvar_33_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_33[i] == 'V' :
				libvar_33_id[i] = 1

		libvar_34 = ('X' * (2 + 15)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 15))
		libvar_34_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_34[i] == 'V' :
				libvar_34_id[i] = 1

		libvar_35 = ('X' * (2 + 11)) + ('V' * (25 - 7)) + ('X' * (50 + 7)) + ('V' * (25 - 7)) + ('X' * (83 + 7 - 7 - 11))
		libvar_35_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_35[i] == 'V' :
				libvar_35_id[i] = 1


		pos_to_libs = [
			#[libvar_2_id, 2],
			#[libvar_8_id, 8],
			#[libvar_5_id, 5],
			#[libvar_11_id, 11],
			#[libvar_20_id, 20],
			#[libvar_21_id, 21],
			[libvar_22_id, 22],
			#[libvar_30_id, 30],
			#[libvar_31_id, 31],
			#[libvar_32_id, 32],
			#[libvar_33_id, 33],
			#[libvar_34_id, 34],
			#[libvar_35_id, 35],
		]
		pos_to_libs_lookup = []
		for pos in range(0, len(pos_to_libs[0][0])) :
			valid_libs = []
			valid_libs_str = ''
			for libvar in pos_to_libs :
				if libvar[0][pos] == 1 :
					valid_libs.append(libvar[1])
					valid_libs_str += '_' + str(libvar[1])
			pos_to_libs_lookup.append([pos, valid_libs, valid_libs_str])


		k_r = numpy.zeros(num_filters)

		f = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/avg_filter_r.txt', 'w')

		meme = open('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/tomtom_meme.txt', 'w')
		meme.write('MEME version 4\n')
		meme.write('\n')
		meme.write('ALPHABET= ACGT')
		meme.write('\n')
		meme.write('strands: + -')
		meme.write('\n')


		mer8 = []
		bases =['A', 'C', 'G', 'T']
		for b1 in bases :
			for b2 in bases :
				for b3 in bases :
					for b4 in bases :
						for b5 in bases :
							for b6 in bases :
								for b7 in bases :
									for b8 in bases :
										mer8.append(b1 + b2 + b3 + b4 + b5+ b6 + b7 + b8)


		#(num_data_points, num_filters, seq_length, 1)
		for k in range(0, num_filters) :


			filter_activations = numpy.concatenate([get_layer0_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((n_original, activation_length))
			#print(filter_activations.shape)
			filter_activations = filter_activations[logodds_test_isinf == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)


			#filter_activations = activations[:, k, :, :].reshape((activations.shape[0], activations.shape[2]))
			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

			max_activation[k, :] = numpy.ravel(numpy.max(filter_activations, axis=1))

			#Region-specific max activations
			max_activation_up[k, :] = numpy.ravel(numpy.max(filter_activations[:,:45], axis=1))
			max_activation_pas[k, :] = numpy.ravel(numpy.max(filter_activations[:,45:52], axis=1))
			max_activation_dn[k, :] = numpy.ravel(numpy.max(filter_activations[:,52:97], axis=1))
			max_activation_comp = numpy.ravel(numpy.max(filter_activations[:,97:], axis=1))

			#pos_activation = numpy.zeros((num_filters, y_test_hat.shape[0], activation_length))
			#pos_activation[k, :, :] = filter_activations[:, :]
			pos_activation = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			if len(spike_index) <= 0 :
				continue

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :]
			filter_L = L_index[spike_index]

			filter_logodds_test = logodds_test[spike_index]
			filter_y_test = y_test[spike_index]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-3000:]#5000
			
			PFM = numpy.zeros((filter_width, self.num_features))
			for ii in range(0, len(top_scoring_index)) :
				i = top_scoring_index[ii]

				filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
				filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :]
				#filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :] * filter_activations[i, max_spike[i]]
				#filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :] * filter_y_test[i]


				#if filter_activations[i, max_spike[i]] >= max_act / 2.0 :
				PFM = PFM + filter_input

			print('Motif ' + str(k))


			#Estimate PPM motif properties
			PPM = numpy.zeros(PFM.shape)
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PPM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])

			meme.write('MOTIF Layer_1_Filter_' + str(k) + '\n')
			meme.write('letter-probability matrix: alength= 4 w= 8 nsites= 3000\n')
			for i in range(0, PPM.shape[0]) :
				for j in range(0, 4) :
					meme.write(' ' + str(round(PPM[i, j], 6)) + ' ')
				meme.write('\n')
			meme.write('\n')

			'''filter_entropy = numpy.zeros(PPM.shape[0])
			for j in range(0, PPM.shape[0]) :
				filter_entropy[j] = -1.0 * numpy.sum(PPM[j, :] * safe_log2(PPM[j, :]))

			joint_prob = numpy.zeros(len(mer8))

			i = 0
			for candidate_mer in mer8 :

				joint_p = 1
				for j in range(0, len(candidate_mer)) :
					candidate_base = candidate_mer[j]

					base_pos = 0
					if candidate_base == 'C' :
						base_pos = 1
					elif candidate_base == 'G' :
						base_pos = 2
					elif candidate_base == 'T' :
						base_pos = 3

					joint_p *= PPM[j, base_pos]

				joint_prob[i] = joint_p

				i += 1

			#mer_sort_index = numpy.argsort(joint_prob)[::-1]
			#joint_prob = joint_prob[mer_sort_index]
			#mer8_sorted = numpy.array(mer8)[mer_sort_index]

			top_n = 10

			trim_start = 0
			trim_end = filter_width
			for j in range(0, filter_width) :
				if filter_entropy[j] <= 1.7 :
					break
				else :
					trim_start = j + 1

			for j in range(filter_width-1, 0, -1) :
				if filter_entropy[j] <= 1.7 :
					break
				else :
					trim_end = j


			trimmed_mer_jointprob_dict = {}
			for i in range(0, len(mer8)) :
				candidate_mer = mer8[i]#[trim_start: trim_end]

				trimmed_mer = ''
				for j in range(0, len(candidate_mer)) :
					if filter_entropy[j] <= 1.7 :
						trimmed_mer += candidate_mer[j]
					else :
						trimmed_mer += ' '
				trimmed_mer = trimmed_mer.strip()

				if trimmed_mer not in trimmed_mer_jointprob_dict :
					trimmed_mer_jointprob_dict[trimmed_mer] = 0
				trimmed_mer_jointprob_dict[trimmed_mer] += joint_prob[i]


			joint_prob = numpy.zeros(len(trimmed_mer_jointprob_dict))
			trimmed_mer = []
			i = 0
			for mer_candidate in trimmed_mer_jointprob_dict :
				joint_prob[i] = trimmed_mer_jointprob_dict[mer_candidate]
				trimmed_mer.append(mer_candidate)

				i += 1

			mer_sort_index = numpy.argsort(joint_prob)[::-1]
			joint_prob = joint_prob[mer_sort_index]
			trimmed_mer_sorted = numpy.array(trimmed_mer)[mer_sort_index]

			mer_logodds_ratio = numpy.zeros(len(trimmed_mer_sorted))

			#mer_logodds_ratio = -1.0 * safe_log2(joint_prob)
			mer_logodds_ratio = numpy.log(joint_prob / (1.0 - joint_prob))
			#for i in range(0, len(trimmed_mer_sorted)) :
			#	mean_else_ratio = numpy.mean(joint_prob[numpy.arange(len(joint_prob)) != i])
			#	mer_logodds_ratio[i] = numpy.log( (joint_prob[i] / (1.0 - joint_prob[i])) / (mean_else_ratio / (1.0 - mean_else_ratio)) )

			
			joint_prob = joint_prob[:top_n][::-1]
			mer_logodds_ratio = mer_logodds_ratio[:top_n][::-1]
			trimmed_mer_sorted = trimmed_mer_sorted[:top_n][::-1]

			mer_labels_joint_prob = []
			mer_labels_logodds_ratio = []
			for i in range(0, len(trimmed_mer_sorted)) :
				mer_labels_joint_prob.append(trimmed_mer_sorted[i] + ' ' + str(round(joint_prob[i], 4)))
				mer_labels_logodds_ratio.append(trimmed_mer_sorted[i] + ' ' + str(round(mer_logodds_ratio[i], 4)))


			fig = plt.figure(figsize=(2, 10))
			plt.pcolor(joint_prob.reshape((len(joint_prob), 1)), cmap=plt.get_cmap('Reds'), vmin=0, vmax=numpy.abs(joint_prob).max())
			plt.xticks([0], [''])
			plt.yticks(numpy.arange(len(joint_prob)) + 0.5, mer_labels_joint_prob)

			plt.axis([0, 1, 0, len(joint_prob)])

			fig.tight_layout()

			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "avg_motif_" + str(k) + "_jointprob.svg")
			plt.close()

			fig = plt.figure(figsize=(2, 10))
			plt.pcolor(mer_logodds_ratio.reshape((len(mer_logodds_ratio), 1)), cmap=plt.get_cmap('Reds'), vmin=mer_logodds_ratio.min(), vmax=mer_logodds_ratio.max())
			plt.xticks([0], [''])
			plt.yticks(numpy.arange(len(mer_labels_logodds_ratio)) + 0.5, mer_labels_logodds_ratio)

			plt.axis([0, 1, 0, len(mer_logodds_ratio)])

			fig.tight_layout()

			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "avg_motif_" + str(k) + "_logoddsratio.svg")
			plt.close()'''





			#Calculate Pearson r
			logodds_test_curr = logodds_test
			logodds_test_avg_curr = logodds_test_avg
			logodds_test_std_curr = logodds_test_std

			max_activation_regions = [
				[numpy.ravel(max_activation[k, :]), 'All region'],
				[numpy.ravel(max_activation_up[k, :]), 'Upstream'],
				[numpy.ravel(max_activation_pas[k, :]), 'PAS'],
				[numpy.ravel(max_activation_dn[k, :]), 'Downstream']
			]
			r_up_k = 0
			r_pas_k = 0
			r_dn_k = 0
			for region in max_activation_regions :
				max_activation_k = region[0]

				max_activation_k = max_activation_k[L_index > 5]
				logodds_test_curr = logodds_test[L_index > 5]

				max_activation_k_avg = numpy.average(max_activation_k)
				max_activation_k_std = numpy.sqrt(numpy.dot(max_activation_k - max_activation_k_avg, max_activation_k - max_activation_k_avg))

				logodds_test_avg_curr = numpy.average(logodds_test_curr)
				logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

				cov = numpy.dot(logodds_test_curr - logodds_test_avg_curr, max_activation_k - max_activation_k_avg)
				r = cov / (max_activation_k_std * logodds_test_std_curr)
				print(region[1] + ' r = ' + str(round(r, 2)))
				f.write('Filter ' + str(k) + ', ' + region[1] + ' r = ' + str(round(r, 2)) + '\n')

			print('')

			prev_selection_libs_str = 'X'
			for pos in range(0, activation_length) :

				pos_activation_curr = pos_activation
				logodds_test_curr = logodds_test
				logodds_test_avg_curr = logodds_test_avg
				logodds_test_std_curr = logodds_test_std
				curr_selection_libs_str = ''
				if pos_to_libs_lookup[pos][2] == '' :
					continue

				pos_to_lib = pos_to_libs_lookup[pos]
				curr_selection_libs_str = pos_to_lib[2]
				if curr_selection_libs_str == prev_selection_libs_str :
					pos_activation_curr = pos_activation_prev
					logodds_test_curr = logodds_test_prev
				else :
					whitelist_index = []
					for i in range(0, len(L_index)) :
						if L_index[i] in pos_to_lib[1] :
							whitelist_index.append(i)
					
					pos_activation_curr = pos_activation[whitelist_index, :]
					logodds_test_curr = logodds_test[whitelist_index]
				logodds_test_avg_curr = numpy.average(logodds_test_curr)
				logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

				if curr_selection_libs_str == '' :
					continue

				pos_activation_k_pos = numpy.ravel(pos_activation_curr[:, pos])
				pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
				pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

				cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k_pos - pos_activation_k_pos_avg)
				r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_std_curr)

				if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
					pos_r[k, pos] = r_k_pos

				prev_selection_libs_str = curr_selection_libs_str
				pos_activation_prev = pos_activation_curr
				logodds_test_prev = logodds_test_curr

			logo_name = "avg_motif_" + str(k) + ".svg"
			logo_name_prob = "avg_motif_" + str(k) + "_prob.svg"
            
			if k == 249 :
				print(PFM)

			logotitle = "Layer 1 Filter " + str(k)
			self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + logo_name, 8, logotitle=logotitle)#r
			#self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + logo_name_prob, 8, logotitle=logotitle, u='probability')#r

		f.close()
		meme.close()


		#All-filter positional Pearson r
		f = plt.figure(figsize=(32, 16))

		avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))
		pos_r = pos_r[avg_r_sort_index, :]

		plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Prox. selection Pearson r for all filters')
		#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
		#xticks = mer_sorted
		plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 185], [0 - 49, 25 - 49, 50 - 49, 75 - 49, 100 - 49, 125 - 49, 150 - 49, 175 - 49, 185 - 49])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

		plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_dropout/deconv/avg_filter/' + "r_pos.svg")
		plt.close()


	def get_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_global_onesided2_dropout/', seq_length=6, normalize=False, logotitle="", u='bits') :
		try :
			if normalize == True :
				for i in range(0, PFM.shape[0]) :
					if numpy.sum(PFM[i, :]) > 0 :
						PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
					#PFM[i, :] *= 10000.0
				#print(PFM)

			#Create weblogo from API
			logo_output_format = "svg"#"png"
			#Load data from an occurence matrix
			data = weblogolib.LogoData.from_counts('ACGT', PFM)

			#Generate color scheme
			colors = weblogolib.ColorScheme([
				weblogolib.SymbolColor("A", "yellow","CFI Binder" ),
				weblogolib.SymbolColor("C", "green","CFI Binder" ),
				weblogolib.SymbolColor("G", "red","CFI Binder" ),
				weblogolib.SymbolColor("T", "blue","CFI Binder" )] )

			#Create options
			options = weblogolib.LogoOptions(fineprint=False,
				logo_title=logotitle,
				color_scheme=colors, 
				stack_width=weblogolib.std_sizes["large"],
				resolution = 400,
				logo_start=1, logo_end=seq_length, stacks_per_line=seq_length, unit_name=u)

			#Create logo
			logo_format = weblogolib.LogoFormat(data, options)

			#Generate image
			formatter = weblogolib.formatters[logo_output_format]
			png = formatter(data, logo_format)

			#Write it
			with open(file_path, "w") as f:
				f.write(png)
		except RuntimeError :
			print('get_logo crashed!')


	def generate_heat_maps(self):
		layer0_left = self.layer0_left
		
		filters = layer0_left.W.eval()
		#(n_kerns, 1, filter_width, 4)

		for k in range(0, filters.shape[0]) :
			kernel = filters[k, 0, :, :].reshape((filters.shape[2], filters.shape[3])).T
			kernel_mean = numpy.mean(kernel, axis=0)

			for j in range(0, kernel.shape[1]) :
				kernel[:, j] -= kernel_mean[j]

			kernel = numpy.fliplr(kernel)

			plt.pcolor(kernel,cmap=cm.RdBu_r)
			plt.colorbar()

			plt.xlabel('Sequence')
			plt.ylabel('Bases')
			plt.title('Kernel Heat Map')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'])#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

			plt.savefig("cnn_motif_analysis/fullseq_global_onesided2_dropout/kernel/kernel" + str(k) + ".png")
			plt.savefig("cnn_motif_analysis/fullseq_global_onesided2_dropout/kernel/kernel" + str(k) + ".svg")
			plt.close()

	def dropit(self, srng, weight, drop):
		# proportion of probability to retain
		retain_prob = 1 - drop
		# a masking variable
		mask = srng.binomial(n=1, p=retain_prob, size=weight.shape, dtype=theano.config.floatX)
		# final weight with dropped neurons
		return T.cast(weight * mask, theano.config.floatX)

	def dont_dropit(self, weight, drop):
		return (1 - drop)*T.cast(weight, theano.config.floatX)

	def dropout_layer(self, srng, weight, drop, train = 1):
		return T.switch(theano.tensor.eq(train, 1), self.dropit(srng, weight, drop), self.dont_dropit(weight, drop))

	def __init__(self, train_set, valid_set, learning_rate=0.1, drop=0, n_epochs=30, nkerns=[30, 40, 50], batch_size=50, num_features=4, randomized_regions=[(2, 37), (45, 80)], load_model=True, train_model_flag=False, store_model=False, layer_1_init_pwms=None, layer_1_knockdown=None, layer_1_knockdown_init='rand', layer_2_knockdown=None, layer_2_knockdown_init='rand', dataset='default', store_as_dataset='default', cell_line='default'):
		numpy.random.seed(23455)
		rng = numpy.random.RandomState(23455)

		srng = RandomStreams(rng.randint(999999))

		orthoInit = None#OrthogonalInit(gain='relu')#None

		#Guided Backprop Gradient, only instantiate once
		#modded_relu = GuidedBackprop(relu)
		#Zeiler Deconv prop
		#modded_relu = ZeilerBackprop(relu)
		#Regular ReLU Backprop
		modded_relu = relu

		self.batch_size = batch_size
		self.randomized_regions = randomized_regions
		
		train_set_x, train_set_y, train_set_L, train_set_d = train_set
		valid_set_x, valid_set_y, valid_set_L, valid_set_d = valid_set

		# compute number of minibatches for training, validation and testing
		n_train_batches = train_set_x.get_value(borrow=True).shape[0]
		n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
		n_train_batches /= batch_size
		n_valid_batches /= batch_size

		# allocate symbolic variables for the data
		index = T.lscalar()  # index to a [mini]batch

		learning_rate_input = T.fscalar()

		# start-snippet-1
		x_left = T.tensor3('x_left')   # the data is presented as rasterized images
		x_right = T.tensor3('x_right')   # the data is presented as rasterized images

		L_input = T.matrix('L_input')
		d_input = T.matrix('d_input')

		deactivated_filter_level1 = T.lscalar()
		deactivated_output_level1 = T.dscalar()
		self.deactivated_filter_level1 = deactivated_filter_level1
		self.deactivated_output_level1 = deactivated_output_level1
		
		y = T.matrix('y')  # the labels are presented as 1D vector of
		#y = T.matrix('y')

		#print(train_set_y.dtype)
		#print(train_set_L.dtype)
		#print(train_set_d.dtype)
		#print(1 + '')


		self.x_left = x_left
		self.x_right = x_right
		self.y = y
		self.L_input = L_input
		self.d_input = d_input
		
		left_random_size = randomized_regions[0][1] - randomized_regions[0][0]
		right_random_size = randomized_regions[1][1] - randomized_regions[1][0]
		
		self.left_random_size = left_random_size
		self.right_random_size = right_random_size
		self.num_features = num_features
		self.input_size = left_random_size + right_random_size
		
		######################
		# BUILD ACTUAL MODEL #
		######################
		print('... building the model')

		# Reshape matrix of rasterized images of shape (batch_size, 28 * 28)
		# to a 4D tensor, compatible with our LeNetConvPoolLayer
		# (150, 4) is the size of MNIST images.
		layer0_input_left = x_left.reshape((batch_size, 1, left_random_size, num_features))
		layer0_input_right = x_right.reshape((batch_size, 1, right_random_size, num_features))

		# Construct the first convolutional pooling layer:
		# filtering reduces the image size to (101-6+1 , 4-4+1) = (96, 1)
		# maxpooling reduces this further to (96/1, 1/1) = (96, 1)
		# 4D output tensor is thus of shape (batch_size, nkerns[0], 96, 1)
		layer0_left = LeNetConvPoolLayer(
			orthoInit,
			rng,
			input=layer0_input_left,
			deactivated_filter=deactivated_filter_level1,
			deactivated_output=deactivated_output_level1,
			image_shape=(batch_size, 1, left_random_size, num_features),
			filter_shape=(nkerns[0], 1, 8, num_features),
			poolsize=(2, 1),#poolsize=(1, 1),
			stride=(1, 1),
			activation_fn=modded_relu#relu
			,load_model = load_model,
			filter_init_pwms=layer_1_init_pwms,
			filter_knockdown=layer_1_knockdown,
			filter_knockdown_init=layer_1_knockdown_init,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv0_left_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv0_left_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv0_left_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv0_left_b'
		)
		
		'''layer0_right = LeNetConvPoolLayer(
			rng,
			input=layer0_input_right,
			image_shape=(batch_size, 1, right_random_size, num_features),
			filter_shape=(nkerns[0], 1, 7, num_features),
			poolsize=(1, 1),
			use_relu=True
			,load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv0_right_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv0_right_b'
		)

		layer1_input = T.concatenate([layer0_left.output, layer0_right.output], axis=2)'''
		
		# Construct the second convolutional pooling layer
		# filtering reduces the image size to (96-5+1, 1-1+1) = (92, 1)
		# maxpooling reduces this further to (92/2, 1/1) = (46, 1)
		# 4D output tensor is thus of shape (batch_size, nkerns[1], 46, 1)
		layer1 = LeNetConvPoolLayer(
			orthoInit,
			rng,
			input=layer0_left.output,
			deactivated_filter=None,
			deactivated_output=None,
			image_shape=(batch_size, nkerns[0], 89, 1),
			filter_shape=(nkerns[1], nkerns[0], 6, 1),
			poolsize=(1, 1),
			activation_fn=modded_relu#relu
			,load_model = load_model,
			filter_knockdown=layer_2_knockdown,
			filter_knockdown_init=layer_2_knockdown_init,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv1_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv1_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv1_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv1_b'
		)
		'''layer1 = LeNetConvPoolLayer(
			rng,
			input=layer0_left.output,
			deactivated_filter=None,
			deactivated_output=None,
			image_shape=(batch_size, nkerns[0], 178, 1),
			filter_shape=(nkerns[1], nkerns[0], 10, 1),
			poolsize=(1, 1),
			activation_fn=modded_relu#relu
			,load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv1_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv1_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv1_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_conv1_b'
		)'''



		'''layer2 = LeNetConvPoolLayer(
			rng,
			input=layer1.output,
			image_shape=(batch_size, nkerns[1], 16, 1),
			filter_shape=(nkerns[2], nkerns[1], 5, 1),
			poolsize=(2, 1),
			use_relu=True
			,load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_conv2_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_conv2_b'
		)'''

		# the HiddenLayer being fully-connected, it operates on 2D matrices of
		# shape (batch_size, num_pixels) (i.e matrix of rasterized images).
		# This will generate a matrix of shape (batch_size, nkerns[1] * 16 * 1),
		# or (500, 50 * 21 * 1) = (500, 800) with the default values.
		
		#layer3_input = layer2.output.flatten(2)
		layer3_input_cnn = layer1.output.flatten(2)
		#layer3_input = layer0_left.output.flatten(2)

		layer3_input = T.concatenate([layer3_input_cnn, d_input], axis=1)

		# construct a fully-connected sigmoidal layer
		layer3 = HiddenLayer(
			rng,
			input=layer3_input,
			n_in=nkerns[1] * (84) * 1 + 1,#n_in=nkerns[1] * (169) * 1 + 1,
			n_out=80,
			#n_out=256,#medium,large
			#n_out=512,#larger
			activation=modded_relu#relu#T.tanh#relu#T.tanh
			,load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_mlp_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_mlp_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_mlp_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_mlp_b',
			orthogonal_init=orthoInit
		)

		layer3_output = layer3.output

		'''if drop != 0 and train_model_flag == True :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 1)
		elif drop != 0 :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 0)'''

		train_drop = T.lscalar()
		if drop != 0 :
			print('Using dropout = ' + str(drop))
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = train_drop)

		layer4_input = T.concatenate([layer3_output, L_input], axis=1)
		#layer4_input = layer3.output

		self.train_drop = train_drop

		# classify the values of the fully-connected sigmoidal layer
		#layer4 = LogisticRegression(input=layer4_input, n_in=80 + 36, n_out=2, load_model = load_model,
		layer4 = LogisticRegression(input=layer4_input, n_in=80 + 36, n_out=2, load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_lr_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_lr_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_lr_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_lr_b')

		self.layer0_left = layer0_left
		#self.layer0_right = layer0_right
		self.layer1 = layer1
		#self.layer2 = layer2
		self.layer3 = layer3
		self.output_layer = layer4
		
		# the cost we minimize during training is the NLL of the model
		cost = layer4.negative_log_likelihood(y)


		# create a function to compute the mistakes that are made by the model
		validate_model = theano.function(
			[index],
			layer4.log_loss(y),
			givens={
				x_left: self.reshape_batch(valid_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
				x_right: self.reshape_batch(valid_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[1][0]:randomized_regions[1][1]],
				y: valid_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: valid_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: valid_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			},
			on_unused_input='ignore'
		)

		validate_rsquare = theano.function(
			[index],
			layer4.rsquare(y),
			givens={
				x_left: self.reshape_batch(valid_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(valid_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: valid_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: valid_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: valid_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			},
			on_unused_input='ignore'
		)
		
		validate_sse = theano.function(
			[index],
			layer4.sse(y),
			givens={
				x_left: self.reshape_batch(valid_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(valid_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: valid_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: valid_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: valid_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			},
			on_unused_input='ignore'
		)
		
		validate_sst = theano.function(
			[index],
			layer4.sst(y),
			givens={
				y: valid_set_y[index * batch_size: (index + 1) * batch_size]
			},
			on_unused_input='ignore'
		)
		
		# create a list of all model parameters to be fit by gradient descent
		params = layer4.params + layer3.params + layer1.params + layer0_left.params
		#params = layer4.params + layer3.params + layer1.params + layer0_left.params# + layer0_right.params
		#params = layer3.params + layer2.params + layer0_left.params + layer0_right.params
		
		# create a list of gradients for all model parameters
		grads = T.grad(cost, params)
		
		# train_model is a function that updates the model parameters by
		# SGD Since this model has many parameters, it would be tedious to
		# manually create an update rule for each model parameter. We thus
		# create the updates list by automatically looping over all
		# (params[i], grads[i]) pairs.
		updates = [
			(param_i, param_i - learning_rate_input * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		train_model = theano.function(
			[index, learning_rate_input],
			cost,
			updates=updates,
			givens={
				x_left: self.reshape_batch(train_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(train_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: train_set_y[index * batch_size: (index + 1) * batch_size],
				L_input: train_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: train_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 1
			},
			on_unused_input='ignore'
		)
		# end-snippet-1
		
		if train_model_flag == True : 
			###############
			# TRAIN MODEL #
			###############
			print('... training')
			# early-stopping parameters
			patience = 140000#140000  # look as this many examples regardless
			patience_increase = 2  # wait this much longer when a new best is
								   # found
			improvement_threshold = 0.998  # a relative improvement of this much is
										   # considered significant
			#validation_frequency = min(n_train_batches, patience / 2)
			validation_frequency = n_train_batches #/ 2
										  # go through this many
										  # minibatche before checking the network
										  # on the validation set; in this case we
										  # check every epoch

			best_validation_loss = numpy.inf
			best_validation_rsquare = 0.
			best_validationfull_rsquare = 0.
			best_iter = 0
			
			best_params = []
			
			start_time = time.clock()

			epoch = 0
			done_looping = False

			while (epoch < n_epochs) and (not done_looping):
				epoch = epoch + 1
				for minibatch_index in xrange(n_train_batches):

					iter = (epoch - 1) * n_train_batches + minibatch_index

					if iter % 100 == 0:
						print('training @ iter = ', iter)
					cost_ij = train_model(minibatch_index, learning_rate)
					
					#print cost_ij

					if (iter + 1) % validation_frequency == 0:

						# compute zero-one loss on validation set
						validation_losses = [validate_model(i) for i
											 in xrange(n_valid_batches)]
						
						validation_rsquares = [validate_rsquare(i) for i
												in xrange(n_valid_batches)]
						
						validation_sses = [validate_sse(i) for i
												in xrange(n_valid_batches)]
						
						validation_ssts = [validate_sst(i) for i
												in xrange(n_valid_batches)]
						
						#print(validation_errors_one)
						#print(validation_errors_zero)

						this_validation_loss = numpy.sum(validation_losses) / (n_valid_batches * batch_size)
						this_validation_rsquare = numpy.mean(validation_rsquares)
						this_validationfull_rsquare = 1.0 - (numpy.sum(validation_sses) / numpy.sum(validation_ssts))

						#print(validation_losses)
						#print(numpy.sum(validation_losses))

						print('epoch %i, minibatch %i/%i, validation logloss %f, mean batch validation R^2 %f %% (total validation R^2 %f %%)' %
							  (epoch, minibatch_index + 1, n_train_batches,
							   this_validation_loss, this_validation_rsquare * 100.0, this_validationfull_rsquare * 100.0))
						print('learning rate = ' + str(round(learning_rate, 6)))

						# if we got the best validation score until now
						if this_validation_loss < best_validation_loss:

							#improve patience if loss improvement is good enough
							if this_validation_loss < best_validation_loss *  \
							   improvement_threshold:
								patience = max(patience, iter * patience_increase)
							#else :
							#	learning_rate /= 2.0

							# save best validation score and iteration number
							best_validation_loss = this_validation_loss
							best_validation_rsquare = this_validation_rsquare
							best_validationfull_rsquare = this_validationfull_rsquare
							best_iter = iter
							
							best_params = [(numpy.array(layer0_left.W.eval(), copy=True), numpy.array(layer0_left.b.eval(), copy=True)),
											#(numpy.array(layer0_right.W.eval(), copy=True), numpy.array(layer0_right.b.eval(), copy=True)),
											(numpy.array(layer1.W.eval(), copy=True), numpy.array(layer1.b.eval(), copy=True)),
											#(numpy.array(layer2.W.eval(), copy=True), numpy.array(layer2.b.eval(), copy=True)),
											(numpy.array(layer3.W.eval(), copy=True), numpy.array(layer3.b.eval(), copy=True)),
											(numpy.array(layer4.W.eval(), copy=True), numpy.array(layer4.b.eval(), copy=True))]

							if store_model == True :
								layer0_left.store_model(best_params[0][0], best_params[0][1])
								#layer0_right.store_model(best_params[1][0], best_params[1][1])
								layer1.store_model(best_params[1][0], best_params[1][1])
								#layer2.store_model(best_params[2][0], best_params[2][1])
								layer3.store_model(best_params[2][0], best_params[2][1])
								layer4.store_model(best_params[3][0], best_params[3][1])
						#else :
						#	learning_rate /= 2.0

					if patience <= iter:
						done_looping = True
						break

			end_time = time.clock()
			
			print('Optimization complete.')
			print('Best validation logloss of %f obtained at iteration %i, '
				  'with mean batch validation R^2 %f %% (total validation R^2 %f %%)' %
				  (best_validation_loss, best_iter + 1, best_validation_rsquare * 100.0, best_validationfull_rsquare * 100.0))
			print >> sys.stderr, ('The code for file ' +
								  os.path.split(__file__)[1] +
								  ' ran for %.2fm' % ((end_time - start_time) / 60.))
			
			if store_model == True :
				layer0_left.store_model(best_params[0][0], best_params[0][1])
				#layer0_right.store_model(best_params[1][0], best_params[1][1])
				layer1.store_model(best_params[1][0], best_params[1][1])
				#layer2.store_model(best_params[2][0], best_params[2][1])
				layer3.store_model(best_params[2][0], best_params[2][1])
				layer4.store_model(best_params[3][0], best_params[3][1])


def get_top_motifs_per_kernel(kernel):
	bases = 'ACGT'
		
	six_mers = []
		
	weights = numpy.zeros(4096)
		
	for i1 in range(0,4):
		for i2 in range(0,4):
			for i3 in range(0,4):
				for i4 in range(0,4):
					for i5 in range(0,4):
						for i6 in range(0,4):
							motif = bases[i1] + bases[i2] + bases[i3] + bases[i4] + bases[i5] + bases[i6]
							six_mers.append(motif)
							weights[i1 * 4**5 + i2 * 4**4 + i3 * 4**3 + i4 * 4**2 + i5 * 4 + i6] = kernel[3-i1,5] + kernel[3-i2,4] + kernel[3-i3,3] + kernel[3-i4,2] + kernel[3-i5,1] + kernel[3-i6,0]

	highest_weight_index = numpy.argsort(weights)[::-1]
	#Pick the 20 first ones of the reversed sorted vector.
	highest_weight_index_top = highest_weight_index[0:50]

	lowest_weight_index = numpy.argsort(weights)
	#Pick the 20 first ones of the sorted vector.
	lowest_weight_index_top = lowest_weight_index[0:50]
		
	return (six_mers, highest_weight_index_top, lowest_weight_index_top, weights)

def get_global_saliency(cnn, test_set) :

	test_set_x, test_set_y, test_set_L = test_set

	saliency = cnn.get_saliency()
	saliency = saliency.reshape((saliency.shape[0] * saliency.shape[1], saliency.shape[2], saliency.shape[3]))
	
	#Scale positive correlation
	pos_saliency = numpy.maximum(0, saliency) / saliency.max(axis=0)
	pos_saliency_index = saliency > 0
	pos_saliency_id = numpy.zeros(pos_saliency.shape)
	pos_saliency_id[pos_saliency_index] = 1

	neg_saliency = numpy.maximum(0, -saliency) / -saliency.min(axis=0)
	neg_saliency_index = saliency < 0
	neg_saliency_id = numpy.zeros(neg_saliency.shape)
	neg_saliency_id[neg_saliency_index] = 1

	cnn.set_data(test_set_x, test_set_y, test_set_L)

	y_test_hat = cnn.get_prediction()
	y_test = test_set_y.eval()[:y_test_hat.shape[0],1]

	s_test_hat = cnn.get_class_score()

	X_test = test_set_x.eval()[:y_test_hat.shape[0],:]

	PFM_pos = numpy.zeros((93, 4))
	PFM_pos_scaled = numpy.zeros((93, 4))
	PFM_neg = numpy.zeros((93, 4))
	PFM_neg_scaled = numpy.zeros((93, 4))
	for i in range(0, y_test_hat.shape[0]) :
		X_point = numpy.ravel(X_test[i,:].todense())
		X_point = X_point.reshape((len(X_point) / 4, 4))

		pos_input = numpy.multiply(X_point, pos_saliency_id[i])
		neg_input = numpy.multiply(X_point, neg_saliency_id[i])

		PFM_pos = PFM_pos + pos_input
		PFM_pos_scaled = PFM_pos_scaled + pos_input * numpy.abs(s_test_hat[i, 1])
		PFM_neg = PFM_neg + neg_input
		PFM_neg_scaled = PFM_neg_scaled + neg_input * numpy.abs(s_test_hat[i, 0])


	logo_name = "pos_unscaled.png"
	cnn.get_logo(0, PFM_pos, 'cnn_motif_analysis/fullseq_v_pad0/deconv/' + logo_name, 93, True)
	logo_name = "pos_scaled.png"
	cnn.get_logo(0, PFM_pos_scaled, 'cnn_motif_analysis/fullseq_v_pad0/deconv/' + logo_name, 93, True)

	logo_name = "neg_unscaled.png"
	cnn.get_logo(0, PFM_neg, 'cnn_motif_analysis/fullseq_v_pad0/deconv/' + logo_name, 93, True)
	logo_name = "neg_scaled.png"
	cnn.get_logo(0, PFM_neg_scaled, 'cnn_motif_analysis/fullseq_v_pad0/deconv/' + logo_name, 93, True)


def read_layer1_init_filter() :

	manual_pwms = numpy.load('pwms_init/manual/pwm_init_manual.npy')

	pwm_manual_tgtct = manual_pwms[0, :, :]
	pwm_manual_ttatt = manual_pwms[1, :, :]
	pwm_manual_ggaggg = manual_pwms[2, :, :]
	pwm_manual_agga = manual_pwms[3, :, :]

	doubledope_upstream_pos_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_upstream_doubledope_pos.npy')
	pwm_doubledope_tgta = doubledope_upstream_pos_pwms[1, :, :]
	pwm_doubledope_tgtca = doubledope_upstream_pos_pwms[2, :, :]
	pwm_doubledope_tgtta = doubledope_upstream_pos_pwms[4, :, :]
	pwm_doubledope_ctgt = doubledope_upstream_pos_pwms[5, :, :]
	pwm_doubledope_tgtga = doubledope_upstream_pos_pwms[10, :, :]

	doubledope_upstream_neg_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_upstream_doubledope_neg.npy')
	pwm_doubledope_taggga = doubledope_upstream_neg_pwms[6, :, :]

	doubledope_pas_pos_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_pas_doubledope_pos.npy')
	pwm_doubledope_pas = doubledope_pas_pos_pwms[0, :, :]

	doubledope_downstream_pos_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_downstream_doubledope_pos.npy')
	pwm_doubledope_ttttt = doubledope_downstream_pos_pwms[1, :, :]
	pwm_doubledope_ttatt = doubledope_downstream_pos_pwms[2, :, :]
	pwm_doubledope_tttgt = doubledope_downstream_pos_pwms[3, :, :]
	pwm_doubledope_tcatt = doubledope_downstream_pos_pwms[4, :, :]
	pwm_doubledope_tgttt = doubledope_downstream_pos_pwms[5, :, :]
	pwm_doubledope_ttctt = doubledope_downstream_pos_pwms[7, :, :]
	pwm_doubledope_tctgt = doubledope_downstream_pos_pwms[9, :, :]
	pwm_doubledope_ttcact = doubledope_downstream_pos_pwms[11, :, :]

	doubledope_downstream_neg_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_downstream_doubledope_neg.npy')
	pwm_doubledope_ggtaag = doubledope_downstream_neg_pwms[0, :, :]
	pwm_doubledope_aggga = doubledope_downstream_neg_pwms[3, :, :]
	pwm_doubledope_ggcgg = doubledope_downstream_neg_pwms[6, :, :]
	pwm_doubledope_ttagg = doubledope_downstream_neg_pwms[7, :, :]
	pwm_doubledope_ggcagg = doubledope_downstream_neg_pwms[8, :, :]
	pwm_doubledope_ggacgg = doubledope_downstream_neg_pwms[10, :, :]

	doubledope_fdownstream_pos_pwms = numpy.load('pwms_init/exp/doubledope/pwm_init_fdownstream_doubledope_pos.npy')
	pwm_doubledope_gggggg = doubledope_fdownstream_pos_pwms[0, :, :]
	pwm_doubledope_gccg = doubledope_fdownstream_pos_pwms[2, :, :]
	pwm_doubledope_cgcgcg = doubledope_fdownstream_pos_pwms[4, :, :]



	tomm5_upstream_pos_pwms = numpy.load('pwms_init/exp/tomm5/pwm_init_upstream_tomm5_pos.npy')
	pwm_tomm5_ttgtc = tomm5_upstream_pos_pwms[1, :, :]
	pwm_tomm5_ccccct = tomm5_upstream_pos_pwms[3, :, :]
	pwm_tomm5_gtct = tomm5_upstream_pos_pwms[5, :, :]
	pwm_tomm5_tatatc = tomm5_upstream_pos_pwms[11, :, :]

	tomm5_upstream_neg_pwms = numpy.load('pwms_init/exp/tomm5/pwm_init_upstream_tomm5_neg.npy')
	pwm_tomm5_gcggc = tomm5_upstream_neg_pwms[1, :, :]
	pwm_tomm5_ctaggt = tomm5_upstream_neg_pwms[11, :, :]

	tomm5_downstream_pos_pwms = numpy.load('pwms_init/exp/tomm5/pwm_init_downstream_tomm5_pos.npy')
	pwm_tomm5_tcatta = tomm5_downstream_pos_pwms[0, :, :]
	pwm_tomm5_tcttta = tomm5_downstream_pos_pwms[1, :, :]
	pwm_tomm5_cgtgt = tomm5_downstream_pos_pwms[3, :, :]
	pwm_tomm5_gttatt = tomm5_downstream_pos_pwms[4, :, :]
	pwm_tomm5_tgattt = tomm5_downstream_pos_pwms[8, :, :]
	pwm_tomm5_gttttt = tomm5_downstream_pos_pwms[10, :, :]

	tomm5_downstream_neg_pwms = numpy.load('pwms_init/exp/tomm5/pwm_init_downstream_tomm5_neg.npy')
	pwm_tomm5_tacagg = tomm5_downstream_neg_pwms[5, :, :]
	pwm_tomm5_tttagg = tomm5_downstream_neg_pwms[6, :, :]
	pwm_tomm5_gggagg = tomm5_downstream_neg_pwms[7, :, :]



	simple_upstream_pos_pwms = numpy.load('pwms_init/exp/simple/pwm_init_upstream_simple_pos.npy')
	pwm_simple_tgtatt = simple_upstream_pos_pwms[1, :, :]
	pwm_simple_ccgttt = simple_upstream_pos_pwms[10, :, :]
	pwm_simple_cttttt = simple_upstream_pos_pwms[11, :, :]

	simple_downstream_pos_pwms = numpy.load('pwms_init/exp/simple/pwm_init_downstream_simple_pos.npy')
	pwm_simple_ttttg = simple_downstream_pos_pwms[0, :, :]
	pwm_simple_tttag = simple_downstream_pos_pwms[1, :, :]
	pwm_simple_tgcgtc = simple_downstream_pos_pwms[9, :, :]
	pwm_simple_ttcatt = simple_downstream_pos_pwms[10, :, :]

	simple_downstream_neg_pwms = numpy.load('pwms_init/exp/simple/pwm_init_downstream_simple_neg.npy')
	pwm_simple_gggat = simple_downstream_neg_pwms[1, :, :]
	pwm_simple_ggcgga = simple_downstream_neg_pwms[6, :, :]
	pwm_simple_ggac = simple_downstream_neg_pwms[10, :, :]

	simple_fdownstream_pos_pwms = numpy.load('pwms_init/exp/simple/pwm_init_fdownstream_simple_pos.npy')
	pwm_simple_cgccg = simple_fdownstream_pos_pwms[2, :, :]
	pwm_simple_aggg = simple_fdownstream_pos_pwms[4, :, :]
	pwm_simple_cgtcg = simple_fdownstream_pos_pwms[5, :, :]

	simple_fdownstream_neg_pwms = numpy.load('pwms_init/exp/simple/pwm_init_fdownstream_simple_neg.npy')
	pwm_simple_ttatt = simple_fdownstream_neg_pwms[1, :, :]
	pwm_simple_ctaatt = simple_fdownstream_neg_pwms[2, :, :]
	pwm_simple_tcatt = simple_fdownstream_neg_pwms[3, :, :]
	pwm_simple_tgatt = simple_fdownstream_neg_pwms[4, :, :]
	pwm_simple_ccaat = simple_fdownstream_neg_pwms[6, :, :]


	init_pwms = numpy.zeros((40, 8, 4))

	#Upstream Init
	init_pwms[0, :, :] = pwm_doubledope_tgta
	init_pwms[1, :, :] = pwm_doubledope_tgtca
	init_pwms[2, :, :] = pwm_doubledope_tgtta
	init_pwms[3, :, :] = pwm_doubledope_tgtga
	init_pwms[4, :, :] = pwm_doubledope_ctgt
	init_pwms[5, :, :] = pwm_simple_cttttt
	init_pwms[6, :, :] = pwm_simple_tgtatt
	init_pwms[7, :, :] = pwm_tomm5_ccccct
	init_pwms[8, :, :] = pwm_tomm5_tatatc
	init_pwms[9, :, :] = pwm_tomm5_gtct

	init_pwms[10, :, :] = pwm_doubledope_taggga
	init_pwms[11, :, :] = pwm_tomm5_ctaggt
	init_pwms[12, :, :] = pwm_tomm5_gcggc
	

	#PAS Init
	init_pwms[13, :, :] = pwm_doubledope_pas

	#Manual Init
	init_pwms[14, :, :] = pwm_manual_tgtct
	init_pwms[15, :, :] = pwm_manual_ggaggg
	init_pwms[16, :, :] = pwm_manual_agga

	#Downstream Init
	init_pwms[17, :, :] = pwm_doubledope_ttttt
	init_pwms[18, :, :] = pwm_doubledope_tgttt
	init_pwms[19, :, :] = pwm_doubledope_ttctt
	init_pwms[20, :, :] = pwm_doubledope_tctgt
	init_pwms[21, :, :] = pwm_simple_ttttg
	init_pwms[22, :, :] = pwm_simple_tcatt
	init_pwms[23, :, :] = pwm_simple_tgatt
	init_pwms[24, :, :] = pwm_simple_ttatt
	init_pwms[25, :, :] = pwm_tomm5_tcatta
	init_pwms[26, :, :] = pwm_tomm5_cgtgt
	init_pwms[27, :, :] = pwm_tomm5_gttatt
	init_pwms[28, :, :] = pwm_tomm5_gttttt

	init_pwms[29, :, :] = pwm_tomm5_tttagg
	init_pwms[30, :, :] = pwm_tomm5_gggagg
	init_pwms[31, :, :] = pwm_tomm5_tacagg
	init_pwms[32, :, :] = pwm_doubledope_gggggg
	init_pwms[33, :, :] = pwm_doubledope_cgcgcg
	init_pwms[34, :, :] = pwm_doubledope_ttagg
	init_pwms[35, :, :] = pwm_doubledope_ggcagg
	init_pwms[36, :, :] = pwm_doubledope_ggacgg
	init_pwms[37, :, :] = pwm_doubledope_ggcgg
	init_pwms[38, :, :] = pwm_doubledope_aggga
	init_pwms[39, :, :] = pwm_doubledope_ggtaag

	for i in range(0, init_pwms.shape[0]) :
		init_pwms[i, :, :] = numpy.fliplr(numpy.flipud(init_pwms[i, :, :]))

	print('Returning flipped init kernels.')

	return init_pwms

def check_layer_weights(cnn) :

	#filter_shape: (number of filters, num input feature maps, filter height, filter width)

	W_layer_1 = numpy.array(cnn.layer0_left.W.eval())
	b_layer_1 = numpy.array(cnn.layer0_left.b.eval())

	for i in range(0, W_layer_1.shape[0]) :
		W_layer_1[i, 0, :, :] = numpy.fliplr(numpy.flipud(W_layer_1[i, 0, :, :]))

	W_layer_2 = numpy.array(cnn.layer1.W.eval())
	b_layer_2 = numpy.array(cnn.layer1.b.eval())

	for i in range(0, W_layer_2.shape[0]) :
		for j in range(0, W_layer_2.shape[1]) :
			W_layer_2[i, j, :, :] = numpy.fliplr(numpy.flipud(W_layer_2[i, j, :, :]))


	layer_2_filter_47_sensitive = []
	for i in range(0, W_layer_2.shape[0]) :
		for j in range(0, W_layer_2.shape[2]) :
			if W_layer_2[i, 47, j, 0] > 0 :
				for k in range(j - 4, j + 4) :
					if k < 0 or k >= W_layer_2.shape[2] :
						continue
					for l in range(0, W_layer_2.shape[1]) :
						if l != 47 and W_layer_2[i, l, k, 0] > 0 :
							print('Layer 2 filter ' + str(i))
							print('Filter 47 fires at position ' + str(j) + ' (' + str(W_layer_2[i, 47, j, 0]) + '), Filter ' + str(l) + ' fires at position ' + str(k) + ' (' + str(W_layer_2[i, l, k, 0]) + ')')


def evaluate_cnn(dataset='general3_antimisprime_orig'):#_pasaligned

	count_filter=0#10#0

	#input_datasets = load_input_data(dataset, shuffle=True, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
	input_datasets = load_input_data(dataset, shuffle=True, shuffle_all=False, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
	
	train_set_x = input_datasets[0]
	valid_set_x = input_datasets[1]
	test_set_x = input_datasets[2]

	shuffle_index = input_datasets[3]

	train_set_L = input_datasets[4]
	valid_set_L = input_datasets[5]
	test_set_L = input_datasets[6]
	
	misprime_index = input_datasets[7]

	train_set_d = input_datasets[8]
	valid_set_d = input_datasets[9]
	test_set_d = input_datasets[10]

	shuffle_all_index = input_datasets[11]

	train_set_c = input_datasets[12]
	valid_set_c = input_datasets[13]
	test_set_c = input_datasets[14]

	#output_datasets = load_output_data(dataset, None, None, [1], shuffle_index, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_index=misprime_index, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
	output_datasets = load_output_data(dataset, None, None, [1], shuffle_index, shuffle_all_index, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_index=misprime_index, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
	
	train_set_y = output_datasets[0]
	valid_set_y = output_datasets[1]
	test_set_y = output_datasets[2]


	#run_name = '_Global_Onesided_DoubleDope_Simple'
	#run_name = '_Global_Onesided_DoubleDope_TOMM5'
	#run_name = '_Global_Onesided_DoubleDope_Simple_TOMM5'
	run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_2'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pasaligned'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pred_32_33_35'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_32_33_35_pred_30_to_35'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_32_33_35_pred_30_31_34'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_All_pred_30_to_35'
	#run_name = '_Global2_Onesided2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_All'
	#run_name = '_Global2_Onesided_DoubleDope_Simple'
	#run_name = '_Global2_Onesided_DoubleDope_Simple_TOMM5'


	print('Train set sublib distribution:')
	L_train = train_set_L.eval()
	L_train_sum = numpy.ravel(numpy.sum(L_train, axis=0))
	for i in range(0, len(L_train_sum)) :
		if L_train_sum[i] > 0 :
			print(str(i) + ": " + str(L_train_sum[i]))

	print('Validation set sublib distribution:')
	L_valid = valid_set_L.eval()
	L_valid_sum = numpy.ravel(numpy.sum(L_valid, axis=0))
	for i in range(0, len(L_valid_sum)) :
		if L_valid_sum[i] > 0 :
			print(str(i) + ": " + str(L_valid_sum[i]))

	print('Test set sublib distribution:')
	L_test = test_set_L.eval()
	L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
	for i in range(0, len(L_test_sum)) :
		if L_test_sum[i] > 0 :
			print(str(i) + ": " + str(L_test_sum[i]))


	#Initialize kernels manually

	'''layer_1_dict = {

		47 : numpy.fliplr(numpy.flipud(numpy.array(
				[
					[-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
					[-0.5, -0.1, -0.5, 0.25, -0.5, 0.01, 0.01, 0.01],
					[-0.5, 0.5, -0.5, 0.25, -0.5, 0.01, 0.01, 0.01],
					[0.5, -0.5, 0.5, -0.5, 0.5, 0.01, 0.01, 0.01]
				])))

	}'''


	init_pwms = read_layer1_init_filter()


	batch_size = 50#50#1#50#1#50
	
	cnn = DualCNN(
		(train_set_x, train_set_y, train_set_L, train_set_d),
		(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
		learning_rate=0.1,
		drop=0.2,
		n_epochs=10,
		#nkerns=[30, 50, 70],#APA_Six_30_31_34_smaller
		#nkerns=[50, 90, 70],#APA_Six_30_31_34_small
		nkerns=[70, 110, 70],#APA_Six_30_31_34
		#nkerns=[96, 128, 70],#APA_Six_All
		#nkerns=[256, 512, 70],#APA_Six_30_31_34_large
		#nkerns=[128, 256, 70],#APA_Six_30_31_34_medium
		#nkerns=[256, 512, 70],#APA_Six_30_31_34_larger
		batch_size=batch_size,
		num_features=4,
		randomized_regions=[(0, 185), (185, 185)],
		load_model=True,
		train_model_flag=False,
		store_model=False,
		layer_1_init_pwms=None,#init_pwms,
		layer_1_knockdown=None,#[1, 3, 4, 9, 15, 30, 37, 40, 42, 59, 62],
		layer_1_knockdown_init='rand',#zeros
		layer_2_knockdown=None,#[19, 44, 69, 87],
		layer_2_knockdown_init='rand',#zeros
		#dataset='general' + 'apa_sparse_general' + '_global_onesided',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided'#'_global_onesided_finetuned_TOMM5'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided_finetuned_TOMM5',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided_finetuned_TOMM5'#'_global_onesided_finetuned_TOMM5'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided_DoubleDope_TOMM5',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided_DoubleDope_TOMM5'#'_global_onesided_finetuned_TOMM5'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2_finetuned_TOMM5_APA_Six_30_31_34'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2prox_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2prox_finetuned_TOMM5_APA_Six_30_31_34'
		
		dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
		store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_32_33_35',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_32_33_35'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_All',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_All'

		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_2',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_2'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_nopool',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_nopool'
		
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_smaller',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_smaller'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_small',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_small'
		
		
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_All',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_All'
		
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_medium',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_medium'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_large',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_large'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_larger',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_larger'
	)
	
	print('Trained sublib bias terms:')
	lrW = cnn.output_layer.W.eval()
	lrW = numpy.ravel(lrW[lrW.shape[0] - 36:, 1])
	for i in range(0, len(lrW)) :
		if lrW[i] != 0 :
			print(str(i) + ": " + str(lrW[i]))
	print('Global bias:')
	print(cnn.output_layer.b.eval())

	#cnn.set_data(test_set_x, test_set_y, test_set_L)

	#cnn.generate_heat_maps()
	#cnn.generate_local_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='pos_and_neg')
	#cnn.generate_global_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='pos')
	#cnn.generate_global_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='neg')
	#cnn.generate_local_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='pos_and_neg')
	#cnn.generate_global_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='pos')
	#cnn.generate_global_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='neg')
	#cnn.generate_sequence_logos((test_set_x, test_set_y, test_set_L, test_set_d))
	#cnn.generate_sequence_logos_level2((test_set_x, test_set_y, test_set_L, test_set_d))

	#get_global_saliency(cnn, (test_set_x, test_set_y, test_set_L))

	#debug_libraries(cnn, test_set_x, test_set_y, test_set_L)
	#print(1 + '')

	#cross_test(cnn)

	#batch_test()

	#store_predictions(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name, test_set_c)
	#split_test(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name)

	#check_layer_weights(cnn)

	print(1 + '')

	cnn.set_data(valid_set_x, valid_set_y, valid_set_L, valid_set_d)

	valid_loss = cnn.get_logloss()
	valid_rsquare = cnn.get_rsquare()
	print(('	Validation set logloss: %f.\n	Validation set R^2: %f %%.') %
		  (valid_loss, valid_rsquare * 100.0))
	y_valid_hat = cnn.get_prediction()
	y_valid = valid_set_y.eval()[:y_valid_hat.shape[0],1]

	area = numpy.pi * (2 * numpy.ones(1))**2

	plt.scatter(y_valid_hat, y_valid, s = area, alpha=0.05)
	plt.plot([0,1], [0,1], '-y')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.title('TOMM5+DoubleDope CNN Prox Ratio (R^2 = ' + str(round(valid_rsquare, 2)) + ')')
	plt.savefig("cnn_validationset_pad0" + run_name + ".png")
	#plt.show()

	logodds_valid_hat = safe_log(y_valid_hat / (1 - y_valid_hat))
	logodds_valid = safe_log(y_valid / (1 - y_valid))
	logodds_valid_isinf = numpy.isinf(logodds_valid)
	logodds_valid_hat = logodds_valid_hat[logodds_valid_isinf == False]
	logodds_valid = logodds_valid[logodds_valid_isinf == False]
	SSE_valid = (logodds_valid - logodds_valid_hat).T.dot(logodds_valid - logodds_valid_hat)
	logodds_valid_average = numpy.average(logodds_valid, axis=0)
	SStot_valid = (logodds_valid - logodds_valid_average).T.dot(logodds_valid - logodds_valid_average)
	logodds_valid_rsquare = 1.0 - (SSE_valid / SStot_valid)

	#Calculate Pearson r
	logodds_valid_hat_avg = numpy.average(logodds_valid_hat)
	logodds_valid_hat_std = numpy.sqrt(numpy.dot(logodds_valid_hat - logodds_valid_hat_avg, logodds_valid_hat - logodds_valid_hat_avg))

	logodds_valid_avg = numpy.average(logodds_valid)
	logodds_valid_std = numpy.sqrt(numpy.dot(logodds_valid - logodds_valid_avg, logodds_valid - logodds_valid_avg))

	cov = numpy.dot(logodds_valid_hat - logodds_valid_hat_avg, logodds_valid - logodds_valid_avg)
	valid_r = cov / (logodds_valid_hat_std * logodds_valid_std)

	print(('	Validation set logodds R^2: %f %%.') %
		  (logodds_valid_rsquare * 100.0))
	print(('	Validation set logodds Pearson r: %f %%.') %
		  (valid_r))

	plt.scatter(logodds_valid_hat, logodds_valid, s = area, alpha=0.05)
	plt.plot([0,1], [0,1], '-y')
	plt.xlim([-4,4])
	plt.ylim([-4,4])
	plt.title('TOMM5+DoubleDope CNN Prox Logodds (R^2 = ' + str(round(logodds_valid_rsquare, 2)) + ', r = ' + str(round(valid_r, 2)) + ')')
	plt.savefig("cnn_logodds_validationset_pad0" + run_name + ".png")
	#plt.show()


	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

	test_loss = cnn.get_logloss()
	test_rsquare = cnn.get_rsquare()
	print(('	Test set logloss: %f.\n	Test set R^2: %f %%.') %
		  (test_loss, test_rsquare * 100.0))
	y_test_hat = cnn.get_prediction()
	y_test = test_set_y.eval()[:y_test_hat.shape[0],1]

	area = numpy.pi * (2 * numpy.ones(1))**2

	plt.scatter(y_test_hat, y_test, s = area, alpha=0.05)
	plt.plot([0,1], [0,1], '-y')
	plt.xlim([0,1])
	plt.ylim([0,1])
	plt.title('TOMM5+DoubleDope CNN Prox Ratio (R^2 = ' + str(round(test_rsquare, 2)) + ')')
	plt.savefig("cnn_testset_pad0" + run_name + ".png")
	#plt.show()

	logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
	logodds_test = safe_log(y_test / (1 - y_test))
	logodds_test_isinf = numpy.isinf(logodds_test)
	logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
	logodds_test = logodds_test[logodds_test_isinf == False]
	SSE_test = (logodds_test - logodds_test_hat).T.dot(logodds_test - logodds_test_hat)
	logodds_test_average = numpy.average(logodds_test, axis=0)
	SStot_test = (logodds_test - logodds_test_average).T.dot(logodds_test - logodds_test_average)
	logodds_test_rsquare = 1.0 - (SSE_test / SStot_test)

	#Calculate Pearson r
	logodds_test_hat_avg = numpy.average(logodds_test_hat)
	logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

	logodds_test_avg = numpy.average(logodds_test)
	logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

	cov = numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test - logodds_test_avg)
	test_r = cov / (logodds_test_hat_std * logodds_test_std)

	print(('	Test set logodds R^2: %f %%.') %
		  (logodds_test_rsquare * 100.0))
	print(('	Test set logodds Pearson r: %f %%.') %
		  (test_r))

	plt.scatter(logodds_test_hat, logodds_test, s = area, alpha=0.05)
	plt.plot([0,1], [0,1], '-y')
	plt.xlim([-4,4])
	plt.ylim([-4,4])
	plt.title('TOMM5+DoubleDope CNN Prox Logodds (R^2 = ' + str(round(logodds_test_rsquare, 2)) + ', r = ' + str(round(test_r, 2)) + ')')
	plt.savefig("cnn_logodds_testset_pad0" + run_name + ".png")
	#plt.show()


def cross_test(cnn) :

	lib_map = [
		['TOMM5', [2, 5, 8, 11], None],
		['DoubleDope', [20], None],
		['Simple', [22], None],
		['ATR', [30], None],
		['AARS', [31], None],
		['HSPE1', [32], None],
		['SNHG6', [33], None],
		['SOX13', [34], None],
		['WHAMMP2', [35], None]
	]

	test_sets = []

	for lib in lib_map :
		lib_name = lib[0]
		L_included = lib[1]

		test_set_size = 10000 * len(L_included)

		count_filter=0

		input_datasets = load_input_data('general3_antimisprime_orig', L_included_list=L_included, constant_test_set_size_param=test_set_size, shuffle=True, shuffle_all=False, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_x = input_datasets[0]
		valid_set_x = input_datasets[1]
		test_set_x = input_datasets[2]

		shuffle_index = input_datasets[3]

		train_set_L = input_datasets[4]
		valid_set_L = input_datasets[5]
		test_set_L = input_datasets[6]
		
		misprime_index = input_datasets[7]

		train_set_d = input_datasets[8]
		valid_set_d = input_datasets[9]
		test_set_d = input_datasets[10]

		shuffle_all_index = input_datasets[11]

		train_set_c = input_datasets[12]
		valid_set_c = input_datasets[13]
		test_set_c = input_datasets[14]

		output_datasets = load_output_data('general3_antimisprime_orig', L_included, test_set_size, [1], shuffle_index, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_index=misprime_index, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_y = output_datasets[0]
		valid_set_y = output_datasets[1]
		test_set_y = output_datasets[2]

		test_sets.append([lib_name, test_set_x, test_set_L, test_set_d, test_set_y, test_set_c])

		run_name = '_Global2_Onesided2AntimisprimeOrig_SingleLibrary_' + lib_name


		print('Train set sublib distribution:')
		L_train = train_set_L.eval()
		L_train_sum = numpy.ravel(numpy.sum(L_train, axis=0))
		for i in range(0, len(L_train_sum)) :
			if L_train_sum[i] > 0 :
				print(str(i) + ": " + str(L_train_sum[i]))

		print('Validation set sublib distribution:')
		L_valid = valid_set_L.eval()
		L_valid_sum = numpy.ravel(numpy.sum(L_valid, axis=0))
		for i in range(0, len(L_valid_sum)) :
			if L_valid_sum[i] > 0 :
				print(str(i) + ": " + str(L_valid_sum[i]))

		print('Test set sublib distribution:')
		L_test = test_set_L.eval()
		L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
		for i in range(0, len(L_test_sum)) :
			if L_test_sum[i] > 0 :
				print(str(i) + ": " + str(L_test_sum[i]))


		batch_size = 50
		
		'''cv_cnn = DualCNN(
			(train_set_x, train_set_y, train_set_L, train_set_d),
			(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
			learning_rate=0.1,
			drop=0.2,
			n_epochs=10,
			nkerns=[70, 110, 70],
			batch_size=batch_size,
			num_features=4,
			randomized_regions=[(0, 185), (185, 185)],
			load_model=False,
			train_model_flag=True,
			store_model=True,
			layer_1_init_pwms=None,
			layer_1_knockdown=None,
			layer_1_knockdown_init='rand',
			layer_2_knockdown=None,
			layer_2_knockdown_init='rand',
			dataset='generalapa_sparse_general_global_onesided2antimisprimeorigdropout_singlelib_' + lib_name,
			store_as_dataset='generalapa_sparse_general_global_onesided2antimisprimeorigdropout_singlelib_' + lib_name
		)'''
		lib[2] = DualCNN(
			(train_set_x, train_set_y, train_set_L, train_set_d),
			(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
			learning_rate=0.1,
			drop=0.2,
			n_epochs=10,
			nkerns=[70, 110, 70],
			batch_size=batch_size,
			num_features=4,
			randomized_regions=[(0, 185), (185, 185)],
			load_model=True,
			train_model_flag=False,
			store_model=False,
			layer_1_init_pwms=None,
			layer_1_knockdown=None,
			layer_1_knockdown_init='rand',
			layer_2_knockdown=None,
			layer_2_knockdown_init='rand',
			dataset='generalapa_sparse_general_global_onesided2antimisprimeorigdropout_singlelib_' + lib_name,
			store_as_dataset='generalapa_sparse_general_global_onesided2antimisprimeorigdropout_singlelib_' + lib_name
		)

	f = open('cross_test_Onesided2AntimisprimeOrigDropout.csv', 'w')

	f.write('model\tlibrary_name\tobserved_logodds\tpredicted_logodds\tcount\n')

	#Evaluate combined model
	for test_set in test_sets :
		lib_name = test_set[0]
		test_set_x = test_set[1]
		test_set_L = test_set[2]
		test_set_d = test_set[3]
		test_set_y = test_set[4]
		test_set_c = test_set[5]
		cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		y_test_hat = numpy.ravel(numpy.array(cnn.get_prediction()))
		y_test = numpy.ravel(numpy.array(test_set_y.eval())[:y_test_hat.shape[0],1])
		c_test = numpy.ravel(numpy.array(test_set_c.eval())[:y_test_hat.shape[0]])
	
		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
		logodds_test = safe_log(y_test / (1 - y_test))
		logodds_test_isinf = numpy.isinf(logodds_test)
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		logodds_test = logodds_test[logodds_test_isinf == False]
		c_test = c_test[logodds_test_isinf == False]

		for i in range(0, len(logodds_test)) :
			f.write('Combined\t' + lib_name + '\t' + str(round(logodds_test[i], 3)) + '\t' + str(round(logodds_test_hat[i], 3)) + '\t' + str(int(c_test[i])) + '\n')

	
	#Evaluate individual models
	for lib in lib_map :
		model_name = lib[0]
		model = lib[2]
		for test_set in test_sets :
			lib_name = test_set[0]
			test_set_x = test_set[1]
			test_set_L = test_set[2]
			test_set_d = test_set[3]
			test_set_y = test_set[4]
			test_set_c = test_set[5]
			model.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

			y_test_hat = numpy.ravel(numpy.array(model.get_prediction()))
			y_test = numpy.ravel(numpy.array(test_set_y.eval())[:y_test_hat.shape[0],1])
			c_test = numpy.ravel(numpy.array(test_set_c.eval())[:y_test_hat.shape[0]])
		
			logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
			logodds_test = safe_log(y_test / (1 - y_test))
			logodds_test_isinf = numpy.isinf(logodds_test)
			logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
			logodds_test = logodds_test[logodds_test_isinf == False]
			c_test = c_test[logodds_test_isinf == False]

			for i in range(0, len(logodds_test)) :
				f.write(model_name + '\t' + lib_name + '\t' + str(round(logodds_test[i], 3)) + '\t' + str(round(logodds_test_hat[i], 3)) + '\t' + str(int(c_test[i])) + '\n')


	f.close()


def batch_test() :

	lib_map = [
		#['TOMM5_APA_Six_30_31_34', [2, 5, 8, 11, 20, 22, 30, 31, 34], [32, 33, 35], True, 30000],
		#['TOMM5_APA_Six_32_33_35', [2, 5, 8, 11, 20, 22, 32, 33, 35], [30, 31, 34], True, 30000],

		#['TOMM5_APA_Six_All', [2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35], [30, 31, 32, 33, 34, 35], True, 60000],

		['TOMM5_APA_Six_31_32_33_34_35', [2, 5, 8, 11, 20, 22, 31, 32, 33, 34, 35], [30], True, 10000],
		['TOMM5_APA_Six_30_32_33_34_35', [2, 5, 8, 11, 20, 22, 30, 32, 33, 34, 35], [31], True, 10000],
		['TOMM5_APA_Six_30_31_33_34_35', [2, 5, 8, 11, 20, 22, 30, 31, 33, 34, 35], [32], True, 10000],
		['TOMM5_APA_Six_30_31_32_34_35', [2, 5, 8, 11, 20, 22, 30, 31, 32, 34, 35], [33], False, 10000],
		['TOMM5_APA_Six_30_31_32_33_35', [2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 35], [34], False, 10000],
		['TOMM5_APA_Six_30_31_32_33_34', [2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34], [35], False, 10000]
	]

	f = open('batch_test_Onesided2AntimisprimeOrigDropout.csv', 'w')

	f.write('model\tlibrary\tobserved_logodds\tpredicted_logodds\tcount\n')

	for lib in lib_map :
		lib_name = lib[0]
		L_included_train = lib[1]
		L_included_test = lib[2]

		already_trained = lib[3]

		test_set_size = lib[4]

		count_filter=0

		input_datasets = load_input_data('general3_antimisprime_orig', L_included_list=L_included_train, constant_test_set_size_param=test_set_size, shuffle=True, shuffle_all=False, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_x = input_datasets[0]
		valid_set_x = input_datasets[1]
		test_set_x = input_datasets[2]

		shuffle_index = input_datasets[3]

		train_set_L = input_datasets[4]
		valid_set_L = input_datasets[5]
		test_set_L = input_datasets[6]
		
		misprime_index = input_datasets[7]

		train_set_d = input_datasets[8]
		valid_set_d = input_datasets[9]
		test_set_d = input_datasets[10]

		shuffle_all_index = input_datasets[11]

		train_set_c = input_datasets[12]
		valid_set_c = input_datasets[13]
		test_set_c = input_datasets[14]

		output_datasets = load_output_data('general3_antimisprime_orig', L_included_train, test_set_size, [1], shuffle_index, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_index=misprime_index, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_y = output_datasets[0]
		valid_set_y = output_datasets[1]
		test_set_y = output_datasets[2]


		print('Evaluating ' + lib_name + '...')


		print('Train set sublib distribution:')
		L_train = train_set_L.eval()
		L_train_sum = numpy.ravel(numpy.sum(L_train, axis=0))
		for i in range(0, len(L_train_sum)) :
			if L_train_sum[i] > 0 :
				print(str(i) + ": " + str(L_train_sum[i]))

		print('Validation set sublib distribution:')
		L_valid = valid_set_L.eval()
		L_valid_sum = numpy.ravel(numpy.sum(L_valid, axis=0))
		for i in range(0, len(L_valid_sum)) :
			if L_valid_sum[i] > 0 :
				print(str(i) + ": " + str(L_valid_sum[i]))

		print('Test set sublib distribution:')
		L_test = test_set_L.eval()
		L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
		for i in range(0, len(L_test_sum)) :
			if L_test_sum[i] > 0 :
				print(str(i) + ": " + str(L_test_sum[i]))


		batch_size = 50

		if already_trained == False :

			print('Training model.')

			cv_cnn = DualCNN(
				(train_set_x, train_set_y, train_set_L, train_set_d),
				(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
				learning_rate=0.1,
				drop=0.2,
				n_epochs=10,
				nkerns=[70, 110, 70],
				batch_size=batch_size,
				num_features=4,
				randomized_regions=[(0, 185), (185, 185)],
				load_model=False,
				train_model_flag=True,
				store_model=True,
				layer_1_init_pwms=None,
				layer_1_knockdown=None,
				layer_1_knockdown_init='rand',
				layer_2_knockdown=None,
				layer_2_knockdown_init='rand',
				dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_' + lib_name,
				store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_' + lib_name
			)


		print('Testing model.')

		input_datasets = load_input_data('general3_antimisprime_orig', L_included_list=L_included_test, constant_test_set_size_param=test_set_size, shuffle=True, shuffle_all=False, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_x = input_datasets[0]
		valid_set_x = input_datasets[1]
		test_set_x = input_datasets[2]

		shuffle_index = input_datasets[3]

		train_set_L = input_datasets[4]
		valid_set_L = input_datasets[5]
		test_set_L = input_datasets[6]
		
		misprime_index = input_datasets[7]

		train_set_d = input_datasets[8]
		valid_set_d = input_datasets[9]
		test_set_d = input_datasets[10]

		shuffle_all_index = input_datasets[11]

		train_set_c = input_datasets[12]
		valid_set_c = input_datasets[13]
		test_set_c = input_datasets[14]

		output_datasets = load_output_data('general3_antimisprime_orig', L_included_test, test_set_size, [1], shuffle_index, balance_data=False, balance_test_set=False, count_filter=count_filter, misprime_index=misprime_index, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=True)
		
		train_set_y = output_datasets[0]
		valid_set_y = output_datasets[1]
		test_set_y = output_datasets[2]

		print('Test set sublib distribution:')
		L_test = test_set_L.eval()
		L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
		for i in range(0, len(L_test_sum)) :
			if L_test_sum[i] > 0 :
				print(str(i) + ": " + str(L_test_sum[i]))

		cv_cnn = DualCNN(
			(train_set_x, train_set_y, train_set_L, train_set_d),
			(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
			learning_rate=0.1,
			drop=0.2,
			n_epochs=10,
			nkerns=[70, 110, 70],
			batch_size=batch_size,
			num_features=4,
			randomized_regions=[(0, 185), (185, 185)],
			load_model=True,
			train_model_flag=False,
			store_model=False,
			layer_1_init_pwms=None,
			layer_1_knockdown=None,
			layer_1_knockdown_init='rand',
			layer_2_knockdown=None,
			layer_2_knockdown_init='rand',
			dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_' + lib_name,
			store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2antimisprimeorigdropout_finetuned_' + lib_name
		)

		cv_cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)
			

		y_test_hat = numpy.ravel(numpy.array(cv_cnn.get_prediction()))
		y_test = numpy.ravel(numpy.array(test_set_y.eval())[:y_test_hat.shape[0],1])
		c_test = numpy.ravel(numpy.array(test_set_c.eval())[:y_test_hat.shape[0]])
		L_test = numpy.ravel(numpy.argmax(numpy.array(test_set_L.eval())[:y_test_hat.shape[0],:], axis=1))
		
		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
		logodds_test = safe_log(y_test / (1 - y_test))
		logodds_test_isinf = numpy.isinf(logodds_test)
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		logodds_test = logodds_test[logodds_test_isinf == False]
		c_test = c_test[logodds_test_isinf == False]
		L_test = L_test[logodds_test_isinf == False]

		for i in range(0, len(logodds_test)) :
			f.write(lib_name + '\t' + str(int(L_test[i])) + '\t' + str(round(logodds_test[i], 3)) + '\t' + str(round(logodds_test_hat[i], 3)) + '\t' + str(int(c_test[i])) + '\n')

	f.close()


def store_predictions(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name, test_set_c) :
	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

	lib_map = [
		[2, 'TOMM52', 'red'],
		[5, 'TOMM55', 'red'],
		[8, 'TOMM58', 'red'],
		[11, 'TOMM511', 'red'],
		[20, 'DoubleDope', 'blue'],
		[22, 'Simple', 'green'],
		[30, 'AARS', 'purple'],
		[31, 'ATR', 'purple'],
		[32, 'HSPE1', 'purple'],
		[33, 'SNHG6', 'purple'],
		[34, 'SOX13', 'purple'],
		[40, 'CRISPR_ATR', 'gold'],
		[41, 'CRISPR_HSPE1', 'gold'],
		[42, 'CRISPR_SNHG6', 'gold'],
		[43, 'CRISPR_SOX13', 'gold'],
	]

	lib_dict = {
		2 : 'TOMM52',
		5 : 'TOMM55',
		8 : 'TOMM58',
		11 : 'TOMM511',
		20 : 'DoubleDope',
		22 : 'Simple',
		30 : 'AARS',
		31 : 'ATR',
		32 : 'HSPE1',
		33 : 'SNHG6',
		34 : 'SOX13',
		35 : 'WHAMMP2',
		40 : 'CRISPR_ATR',
		41 : 'CRISPR_HSPE1',
		42 : 'CRISPR_SNHG6',
		43 : 'CRISPR_SOX13'
	}


	y_test_hat = numpy.ravel(numpy.array(cnn.get_prediction()))
	y_test = numpy.ravel(numpy.array(test_set_y.eval())[:y_test_hat.shape[0],1])
	L_test = numpy.array(test_set_L.eval())[:y_test_hat.shape[0],:]
	c_test = numpy.ravel(numpy.array(test_set_c.eval()))[:y_test_hat.shape[0]]
	X_test = numpy.array(test_set_x.eval().todense())[:y_test_hat.shape[0], :]
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] / 4, 4))

	logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
	logodds_test = safe_log(y_test / (1 - y_test))
	logodds_test_isinf = numpy.isinf(logodds_test)
	logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
	logodds_test = logodds_test[logodds_test_isinf == False]
	y_test_hat = y_test_hat[logodds_test_isinf == False]
	y_test = y_test[logodds_test_isinf == False]
	c_test = c_test[logodds_test_isinf == False]
	X_test = X_test[logodds_test_isinf == False, :]

	L_test = L_test[logodds_test_isinf == False,:]
	L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
	L_test = numpy.ravel(numpy.argmax(L_test, axis=1))


	with open('test_predictions_' + run_name + '.csv', 'w') as f :
		f.write('seq\tlibrary\tlibrary_name\tobserved_logodds\tpredicted_logodds\tcount\n')

		for i in range(0, X_test.shape[0]) :
			seq = translate_matrix_to_seq(X_test[i, :, :])

			li = L_test[i]
			lib_name = lib_dict[li]

			f.write(seq + '\t' + str(int(li)) + '\t' + lib_name + '\t' + str(round(logodds_test[i], 3)) + '\t' + str(round(logodds_test_hat[i], 3)) + '\t' + str(int(c_test[i])) + '\n')

def split_test(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name) :
	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

	lib_map = [
		[2, 'TOMM52', 'red'],
		[5, 'TOMM55', 'red'],
		[8, 'TOMM58', 'red'],
		[11, 'TOMM511', 'red'],
		[20, 'DoubleDope', 'blue'],
		[22, 'Simple', 'green'],
		[30, 'AARS', 'purple'],
		[31, 'ATR', 'purple'],
		[32, 'HSPE1', 'purple'],
		[33, 'SNHG6', 'purple'],
		[34, 'SOX13', 'purple'],
		[35, 'WHAMMP2', 'purple'],
	]


	y_test_hat = cnn.get_prediction()
	y_test = test_set_y.eval()[:y_test_hat.shape[0],1]
	L_test = test_set_L.eval()[:y_test_hat.shape[0],:]

	logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
	logodds_test = safe_log(y_test / (1 - y_test))
	logodds_test_isinf = numpy.isinf(logodds_test)
	logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
	logodds_test = logodds_test[logodds_test_isinf == False]
	y_test_hat = y_test_hat[logodds_test_isinf == False]
	y_test = y_test[logodds_test_isinf == False]

	L_test = L_test[logodds_test_isinf == False,:]
	L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
	L_test = numpy.ravel(numpy.argmax(L_test, axis=1))


	for i in range(0, len(lib_map)) :
		lib = lib_map[i]
		lib_index = lib[0]
		lib_name = lib[1]
		lib_color = lib[2]

		if L_test_sum[lib_index] <= 0 :
			continue

		y_test_hat_curr = y_test_hat[L_test == lib_index]
		y_test_curr = y_test[L_test == lib_index]
		logodds_test_hat_curr = logodds_test_hat[L_test == lib_index]
		logodds_test_curr = logodds_test[L_test == lib_index]


		#Calculate Pearson r
		logodds_test_hat_avg = numpy.average(logodds_test_hat_curr)
		logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

		logodds_test_avg = numpy.average(logodds_test_curr)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

		cov = numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
		test_r = cov / (logodds_test_hat_std * logodds_test_std)

		test_rsquare = test_r * test_r

		f = plt.figure(figsize=(8, 8))

		plt.scatter(logodds_test_hat_curr, logodds_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color=lib_color)
		plt.plot([
			min(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr)),
			max(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
		], alpha=0.5, color='yellow')
			
		plt.axis([numpy.min(logodds_test_hat_curr), numpy.max(logodds_test_hat_curr), numpy.min(logodds_test_curr), numpy.max(logodds_test_curr)])
		plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=24)
		#plt.savefig("cnn_test" + run_name + "_" + lib_name + ".png")
		plt.savefig("cnn_test" + run_name + "_" + lib_name + ".svg")
		#plt.show()
		plt.close()

		f = plt.figure(figsize=(8, 8))

		plt.scatter(logodds_test_hat_curr, logodds_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='black')
		plt.plot([
			min(numpy.min(logodds_test_hat_curr), numpy.min(logodds_test_curr)),
			max(numpy.max(logodds_test_hat_curr), numpy.max(logodds_test_curr))
		], alpha=0.5, color='yellow')
			
		plt.axis([numpy.min(logodds_test_hat_curr), numpy.max(logodds_test_hat_curr), numpy.min(logodds_test_curr), numpy.max(logodds_test_curr)])
		plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=24)
		#plt.savefig("cnn_test" + run_name + "_" + lib_name + "_black.png")
		plt.savefig("cnn_test" + run_name + "_" + lib_name + "_black.svg")
		#plt.show()
		plt.close()
		

	#Calculate Pearson r
	logodds_test_hat_avg = numpy.average(logodds_test_hat)
	logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

	logodds_test_avg = numpy.average(logodds_test)
	logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

	cov = numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test - logodds_test_avg)
	test_r = cov / (logodds_test_hat_std * logodds_test_std)

	test_rsquare = test_r * test_r

	f = plt.figure(figsize=(8, 8))

	plt.scatter(logodds_test_hat, logodds_test, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='black')
	plt.plot([
		min(numpy.min(logodds_test_hat), numpy.min(logodds_test)),
		max(numpy.max(logodds_test_hat), numpy.max(logodds_test))
	], alpha=0.5, color='yellow')
			
	plt.axis([numpy.min(logodds_test_hat), numpy.max(logodds_test_hat), numpy.min(logodds_test), numpy.max(logodds_test)])
	plt.title('R^2 = ' + str(round(test_rsquare, 2)), fontsize=24)
	#plt.savefig("cnn_test" + run_name + "_total_black.png")
	plt.savefig("cnn_test" + run_name + "_total_black.svg")
	#plt.show()
	plt.close()

def safe_log(x, minval=0.02):
    return numpy.log(x.clip(min=minval))

def safe_log2(x):
    x_log = numpy.log2(x)
    x_log[(numpy.isnan(x_log)) | (numpy.isinf(x_log))] = 0
    return x_log

def translate_to_saliency_seq(X_point, input_seq) :
	seq = ""
	for j in range(0, X_point.shape[0]) :
		if input_seq[j] == 'A' and X_point[j, 0] > 0 :
			seq += "A"
		elif input_seq[j] == 'C' and X_point[j, 1] > 0 :
			seq += "C"
		elif input_seq[j] == 'G' and X_point[j, 2] > 0 :
			seq += "G"
		elif input_seq[j] == 'T' and X_point[j, 3] > 0 :
			seq += "T"
		else :
			seq += '.'
	return seq

def translate_matrix_to_seq(X_point) :
	seq = ""
	for j in range(0, X_point.shape[0]) :
		if X_point[j, 0] > 0 :
			seq += "A"
		elif X_point[j, 1] > 0 :
			seq += "C"
		elif X_point[j, 2] > 0 :
			seq += "G"
		elif X_point[j, 3] > 0 :
			seq += "T"
		else :
			seq += "X"
	return seq

def translate_seq_to_matrix(seq) :
	X_point = numpy.zeros((len(seq), 4))
	for j in range(0, X_point.shape[0]) :
		if seq[j] == 'A' :
			X_point[j, 0] = 1
		elif seq[j] == 'C' :
			X_point[j, 1] = 1
		elif seq[j] == 'G' :
			X_point[j, 2] = 1
		elif seq[j] == 'T' :
			X_point[j, 3] = 1
	return X_point

def translate_to_seq(x) :
	X_point = numpy.ravel(x.todense())
	X_point = X_point.reshape((len(X_point) / 4, 4))
	
	seq = ""
	for j in range(0, X_point.shape[0]) :
		if X_point[j, 0] == 1 :
			seq += "A"
		elif X_point[j, 1] == 1 :
			seq += "C"
		elif X_point[j, 2] == 1 :
			seq += "G"
		elif X_point[j, 3] == 1 :
			seq += "T"
		else :
			seq += "X"
	return seq

if __name__ == '__main__':
	evaluate_cnn('general3_antimisprime_orig')#_pasaligned
