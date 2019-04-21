
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

import numpy
import math

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import theano.sparse as Tsparse

from logistic_sgd_global_onesided2_cuts import LogisticRegression, load_input_data, load_output_data
from mlp import HiddenLayer

#import pylab as pl
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import scipy.sparse as sp
import scipy.io as spio

import weblogolib

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import pandas as pd

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
	
	def __init__(self, rng, input, deactivated_filter, deactivated_output, filter_shape, image_shape, poolsize=(2, 2), stride=(1, 1), activation_fn=T.tanh, load_model = False, w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
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
			self.W = theano.shared(
				numpy.asarray(
					rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
					dtype=theano.config.floatX
				),
				borrow=True
			)

			# the bias is a 1D tensor -- one bias per output feature map
			b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
			self.b = theano.shared(value=b_values, borrow=True)
		else :
			self.W = theano.shared(value=self.load_w(w_file + '.npy'), name='W', borrow=True)
			self.b = theano.shared(value=self.load_b(b_file + '.npy'), name='b', borrow=True)

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

		self.conv_out = conv_out
		self.activation = activation

		# add the bias term. Since the bias is a vector (1D array), we first
		# reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
		# thus be broadcasted across mini-batches and feature map
		# width & height
		
		self.output = pooled_out
		
		# store parameters of this layer
		self.params = [self.W, self.b]

def relu(x):
    return T.switch(x<0, 0, x)

class DualCNN(object):

	def set_saliency_functions(self, data_set):
		data_set_x, data_set_y, data_set_L = data_set

		index = T.lscalar()
		batch_size = self.batch_size
		
		self.n_batches = data_set_x.get_value(borrow=True).shape[0] / batch_size
		randomized_regions = self.randomized_regions
		
		x_left = self.x_left
		x_right = self.x_right
		y = self.y
		L_input = self.L_input

		outp = self.output_layer.s_y_given_x
		max_outp = outp[:,1]#T.max(outp, axis=1)
		input_saliency_from_output = theano.grad(max_outp.sum(), wrt=x_left)

		self.compute_input_saliency_from_output = theano.function(
			[index],
			[input_saliency_from_output],
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :]
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
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :]
			}
		)

		self.compute_conv1_saliency_from_output = theano.function(
			[index],
			[conv1_saliency_from_output],
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :]
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

		positions = T.lvector()
		
		self.compute_logloss = theano.function(
			[index],
			self.output_layer.log_loss(y, L_input),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			}
		)
		self.compute_rsquare = theano.function(
			[index],
			self.output_layer.rsquare(y, L_input),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :]
			}
		)
		self.compute_abs_error = theano.function(
			[index],
			self.output_layer.abs_error(y),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
		self.targets = theano.function(
			[index],
			self.output_layer.recall_y(y),
			givens={
				y: Tsparse.basic.dense_from_sparse(data_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :]
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

		self.predict_distribution = theano.function(
			[index, positions],
			self.output_layer.predict_distribution(positions),
			givens={
				x_left: self.reshape_batch(data_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(data_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				L_input: data_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: data_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			},
			on_unused_input='ignore'
		)
		
		data_x = T.dtensor3('x')
		data_L = T.dmatrix('L_i')
		data_d = T.dmatrix('d_i')
		self.online_predict = theano.function(
			[data_x, data_L, data_d],
			self.output_layer.recall(),
			givens={
				x_left: data_x[:,randomized_regions[0][0]:randomized_regions[0][1]].astype(theano.config.floatX),
				x_right: data_x[:,randomized_regions[1][0]:randomized_regions[1][1]].astype(theano.config.floatX),
				L_input: data_L[:, :].astype(theano.config.floatX).astype(theano.config.floatX),
				d_input: data_d[:, :].astype(theano.config.floatX)
				,train_drop: 0
			},
			on_unused_input='ignore'
		)

		data_x = T.dtensor3('x')
		data_L = T.dmatrix('L_i')
		data_d = T.dmatrix('d_i')
		self.online_predict_distribution = theano.function(
			[data_x, data_L, data_d, positions],
			self.output_layer.predict_distribution(positions),
			givens={
				x_left: data_x[:,randomized_regions[0][0]:randomized_regions[0][1]].astype(theano.config.floatX),
				x_right: data_x[:,randomized_regions[1][0]:randomized_regions[1][1]].astype(theano.config.floatX),
				L_input: data_L[:, :].astype(theano.config.floatX),
				d_input: data_d[:, :].astype(theano.config.floatX)
				,train_drop: 0
			},
			on_unused_input='ignore'
		)

	def get_prediction(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.predict(i) for i in xrange(self.n_batches)])
		else:
			return self.predict(i)

	def get_prediction_distrib(self, positions, i=-1):
		if i == -1:
			return numpy.concatenate([self.predict_distribution(i, positions) for i in xrange(self.n_batches)], axis=0)
		else:
			return self.predict_distribution(i, positions)

	def get_target(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.targets(i) for i in xrange(self.n_batches)])
		else:
			return self.targets(i)

	def get_prediction_avgcut(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.predict_avgcut(i) for i in xrange(self.n_batches)])
		else:
			return self.predict_avgcut(i)

	def get_target_avgcut(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.targets_avgcut(i) for i in xrange(self.n_batches)])
		else:
			return self.targets_avgcut(i)

	def get_class_score(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.class_score(i) for i in xrange(self.n_batches)])
		else:
			return self.class_score(i)

	def get_online_prediction(self, data_x):
		return self.online_predict(data_x)

	def get_online_prediction_distrib(self, data_x, data_L, data_d, positions):
		return self.online_predict_distribution(data_x, data_L, data_d, positions)
	
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

	def generate_sequence_logos_long_cut_level2(self, test_set, cut_start, cut_end, name_prefix):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer1 = self.layer1

		positions = numpy.arange(0, 186).tolist()

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

		num_filters = 110

		activation_length = 84

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

		y_test_hat = numpy.array(self.get_prediction_distrib(positions))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		norm_cut_start = 55
		norm_cut_end = 85

		laplace = 0.001		

		y_test_hat_norm = numpy.ravel(numpy.sum(y_test_hat[:, norm_cut_start:norm_cut_end+1] + laplace, axis=1))
		y_test_hat = numpy.ravel(numpy.sum(y_test_hat[:, cut_start:cut_end+1] + laplace, axis=1))

		y_test_hat = y_test_hat / y_test_hat_norm

		y_test_hat_isnan = y_test_hat_norm <= 0.0

		y_test_hat = y_test_hat[y_test_hat_isnan == False]
		input_x = input_x[y_test_hat_isnan == False, :]
		L_index = L_index[y_test_hat_isnan == False]

		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))

		logodds_test_isinf = numpy.isinf(logodds_test_hat)
		y_test_hat = y_test_hat[logodds_test_isinf == False]
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		input_x = input_x[logodds_test_isinf == False, :]
		L_index = L_index[logodds_test_isinf == False]


		logodds_test_hat_avg = numpy.average(logodds_test_hat)
		logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

		print('Reshaped and filtered activations')

		valid_testset_size = y_test_hat.shape[0] - len(numpy.nonzero(y_test_hat_isnan)[0])

		filter_width = 19

		pos_r = numpy.zeros((num_filters, activation_length))


		#valid_activations = numpy.zeros(activations.shape)
		
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

		meme = open('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts_level2/tomtom_meme.txt', 'w')
		meme.write('MEME version 4\n')
		meme.write('\n')
		meme.write('ALPHABET= ACGT')
		meme.write('\n')
		meme.write('strands: + -')
		meme.write('\n')

		for k in range(0, num_filters) :
			filter_activations = numpy.concatenate([get_layer1_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((y_test_hat.shape[0], activation_length))
			filter_activations = filter_activations[y_test_hat_isnan == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)

			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))
			pos_activation = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :]
			filter_L = L_index[spike_index]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-300:]#5000

			PFM = numpy.zeros((filter_width, self.num_features))
			for ii in range(0, len(top_scoring_index)) :
				i = top_scoring_index[ii]

				filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
				filter_input = filter_input[2*max_spike[i]:2*max_spike[i]+filter_width, :]

				PFM = PFM + filter_input

			print('Motif ' + str(k))

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

			for pos in range(0, activation_length) :

				pos_activation_k_pos = numpy.ravel(pos_activation[:, pos])
				pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
				pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

				cov_pos = numpy.dot(logodds_test_hat - logodds_test_hat_avg, pos_activation_k_pos - pos_activation_k_pos_avg)
				r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_hat_std)

				if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
					pos_r[k, pos] = r_k_pos

			logo_name = "avg_motif_" + str(k) + ".png"

			logotitle = "Layer 2 Filter " + str(k)
			self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts_level2/' + logo_name, 19, logotitle=logotitle)#r

		meme.close()

		#All-filter positional Pearson r
		f = plt.figure(figsize=(32, 16))

		avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))

		pos_r = pos_r[avg_r_sort_index, :]

		plt.pcolor(numpy.repeat(pos_r, 2, axis=1),cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Cutsite Pearson r heatmap')
		
		plt.xticks([0, 49, 55, cut_start, cut_end, activation_length * 2], ['-49', '0', '6', 'Cut Start', 'Cut End', str(activation_length * 2 - 49)])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

		plt.axis([0, pos_r.shape[1] * 2, 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts_level2/' + "r_pos_projected_" + name_prefix + ".png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts_level2/' + "r_pos_projected_" + name_prefix + ".svg")
		plt.close()

	def generate_sequence_logos_long_cut(self, test_set, cut_start, cut_end, name_prefix):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer0_left = self.layer0_left

		positions = numpy.arange(0, 186).tolist()

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

		num_filters = 70

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

		print('Computing layer activations')

		y_test_hat = numpy.array(self.get_prediction_distrib(positions))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		norm_cut_start = 55
		norm_cut_end = 95#85

		laplace = 0.001		

		y_test_hat_norm = numpy.ravel(numpy.sum(y_test_hat[:, norm_cut_start:norm_cut_end+1] + laplace, axis=1))
		y_test_hat = numpy.ravel(numpy.sum(y_test_hat[:, cut_start:cut_end+1] + laplace, axis=1))

		y_test_hat = y_test_hat / y_test_hat_norm

		y_test_hat_isnan = y_test_hat_norm <= 0.0

		y_test_hat = y_test_hat[y_test_hat_isnan == False]
		input_x = input_x[y_test_hat_isnan == False, :]
		L_index = L_index[y_test_hat_isnan == False]

		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))

		logodds_test_isinf = numpy.isinf(logodds_test_hat)
		y_test_hat = y_test_hat[logodds_test_isinf == False]
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		input_x = input_x[logodds_test_isinf == False, :]
		L_index = L_index[logodds_test_isinf == False]


		logodds_test_hat_avg = numpy.average(logodds_test_hat)
		logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

		print('Reshaped and filtered activations')

		valid_testset_size = y_test_hat.shape[0] - len(numpy.nonzero(y_test_hat_isnan)[0])

		#APA_SYM_PRX
		libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
		libvar_20_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_20[i] == 'V' :
				libvar_20_id[i] = 1

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

		pos_r = numpy.zeros((num_filters, activation_length))

		filter_width = 8

		for k in range(0, num_filters) :


			filter_activations = numpy.concatenate([get_layer0_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((y_test_hat.shape[0], activation_length))
			filter_activations = filter_activations[y_test_hat_isnan == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)

			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))
			pos_activation = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :]
			filter_L = L_index[spike_index]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-5000:]#5000

			PFM = numpy.zeros((filter_width, self.num_features))
			for ii in range(0, len(top_scoring_index)) :
				i = top_scoring_index[ii]

				filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
				filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :]

				PFM = PFM + filter_input

			print('Motif ' + str(k))


			for pos in range(0, activation_length) :

				pos_activation_k_pos = numpy.ravel(pos_activation[:, pos])
				pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
				pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

				cov_pos = numpy.dot(logodds_test_hat - logodds_test_hat_avg, pos_activation_k_pos - pos_activation_k_pos_avg)
				r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_hat_std)

				if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
					pos_r[k, pos] = r_k_pos


		#All-filter positional Pearson r
		f = plt.figure(figsize=(32, 16))

		avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))

		pos_r = pos_r[avg_r_sort_index, :]

		plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Cutsite Pearson r heatmap')
		
		plt.xticks([0, 49, 55, cut_start, cut_end, activation_length], ['-49', '0', '6', 'Cut Start', 'Cut End', str(activation_length-49)])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

		plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "r_pos_" + name_prefix + ".png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "r_pos_" + name_prefix + ".svg")
		plt.close()

	def generate_motif_cut_heatmap(self, test_set):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer0_left = self.layer0_left

		positions = numpy.arange(0, 186).tolist()

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

		num_filters = 70

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

		print('Computing layer activations')

		y_test_hat = numpy.array(self.get_prediction_distrib(positions))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		norm_cut_start = 55
		norm_cut_end = 95#85

		laplace = 0.001		

		y_test_hat_norm = numpy.ravel(numpy.sum(y_test_hat[:, norm_cut_start:norm_cut_end+1] + laplace, axis=1)).reshape((y_test_hat.shape[0], 1))
		y_test_hat_norm = numpy.tile(y_test_hat_norm, (1, norm_cut_end+1-norm_cut_start))

		y_test_hat = y_test_hat[:, norm_cut_start:norm_cut_end+1] + laplace
		y_test_hat = y_test_hat / y_test_hat_norm
		#y_test_hat = y_test_hat[:, norm_cut_start:norm_cut_end+1]

		y_test_hat_isnan = numpy.ravel(y_test_hat_norm[:, 0]) <= 0.0

		y_test_hat = y_test_hat[y_test_hat_isnan == False, :]
		input_x = input_x[y_test_hat_isnan == False, :]
		L_index = L_index[y_test_hat_isnan == False]

		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))

		logodds_test_isinf = numpy.isinf(numpy.ravel(logodds_test_hat[:, 0]))
		y_test_hat = y_test_hat[logodds_test_isinf == False, :]
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		input_x = input_x[logodds_test_isinf == False, :]
		L_index = L_index[logodds_test_isinf == False]


		#logodds_test_hat_avg = numpy.average(logodds_test_hat)
		#logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

		print('Reshaped and filtered activations')

		valid_testset_size = y_test_hat.shape[0] - len(numpy.nonzero(y_test_hat_isnan)[0])

		#APA_SYM_PRX
		libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
		libvar_20_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_20[i] == 'V' :
				libvar_20_id[i] = 1

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

		pos_r = numpy.zeros((num_filters, activation_length))

		filter_width = 8

		for k in range(0, num_filters) :


			filter_activations = numpy.concatenate([get_layer0_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((y_test_hat.shape[0], activation_length))
			filter_activations = filter_activations[y_test_hat_isnan == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)

			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))
			pos_activation = filter_activations[:, :]



			#spike_index = numpy.nonzero(total_activations > 0)[0]

			#filter_activations = filter_activations[spike_index, :]

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-10000:]#5000


			pos_r = numpy.zeros((activation_length, y_test_hat.shape[1]))

			for cut_pos in range(0, y_test_hat.shape[1]) :
				logodds_test_hat_cut = numpy.ravel(logodds_test_hat[:, cut_pos])
				#logodds_test_hat_cut = numpy.ravel(logodds_test_hat[top_scoring_index, cut_pos])

				logodds_test_hat_cut_avg = numpy.average(logodds_test_hat_cut)
				logodds_test_hat_cut_std = numpy.sqrt(numpy.dot(logodds_test_hat_cut - logodds_test_hat_cut_avg, logodds_test_hat_cut - logodds_test_hat_cut_avg))

				for pos in range(0, activation_length) :
					pos_activation_k_pos = numpy.ravel(pos_activation[:, pos])
					#pos_activation_k_pos = numpy.ravel(pos_activation[top_scoring_index, pos])
					pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
					pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

					cov_pos = numpy.dot(logodds_test_hat_cut - logodds_test_hat_cut_avg, pos_activation_k_pos - pos_activation_k_pos_avg)
					r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_hat_cut_std)

					if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
						pos_r[pos, cut_pos] = r_k_pos

			#All-filter positional Pearson r
			f = plt.figure(figsize=(16, 16))

			plt.pcolor(pos_r.T,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())

			plt.plot([0, activation_length], [0-norm_cut_start, 0-norm_cut_start+activation_length], color='black', linestyle='--')

			plt.colorbar()



			plt.xlabel('Filter position')
			plt.ylabel('Cleavege site position')
			plt.title('Filter-Cutsite Pearson r heatmap')
			
			plt.xticks([0, 49, 55, activation_length], ['-49', '0', '6', str(activation_length-49)])
			plt.yticks(numpy.arange(y_test_hat.shape[1]) + 0.5, numpy.arange(y_test_hat.shape[1]) + norm_cut_start - 49)

			plt.axis([0, pos_r.shape[0], 0, pos_r.shape[1]])

			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "filter_" + str(k) + "_r_cut_pos_.png")
			plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "filter_" + str(k) + "_r_cut_pos_.svg")
			plt.close()
		

	def generate_sequence_logos_cuts(self, test_set, pos_left, pos_right):
		test_set_x, test_set_y, test_set_L, test_set_d = test_set
		self.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

		layer0_left = self.layer0_left

		positions = numpy.arange(0, 186).tolist()

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

		num_filters = 70

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

		print('Computing layer activations')

		y_test_hat = numpy.array(self.get_prediction_distrib(positions))

		input_x = input_x[:y_test_hat.shape[0],:]
		L_index = L_index[:y_test_hat.shape[0]]

		cut_start = 60
		cut_end = 85

		laplace = 0.001

		y_test_hat = y_test_hat[:, cut_start:cut_end+1] + laplace

		y_test_hat_norm = numpy.sum(y_test_hat, axis=1)
		y_test_hat_norm_mat = numpy.zeros((y_test_hat.shape[0], cut_end+1 - cut_start))
		for j in range(0, y_test_hat_norm_mat.shape[1]) :
			y_test_hat_norm_mat[:, j] = y_test_hat_norm


		y_test_hat = y_test_hat / y_test_hat_norm_mat

		y_test_hat_isnan = numpy.sum(y_test_hat, axis=1) <= 0.0

		y_test_hat = y_test_hat[y_test_hat_isnan == False, :]
		input_x = input_x[y_test_hat_isnan == False, :]
		L_index = L_index[y_test_hat_isnan == False]

		print('Reshaped and filtered activations')

		valid_testset_size = y_test_hat.shape[0] - len(numpy.nonzero(y_test_hat_isnan)[0])

		#APA_SYM_PRX
		libvar_20 = ('X' * (24)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (5 + 7 - 7))
		libvar_20_id = numpy.zeros(185 - 7)
		for i in range(0, 185 - 7) :
			if libvar_20[i] == 'V' :
				libvar_20_id[i] = 1

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

		#pos_r = numpy.zeros((num_filters, activation_length))
		pos_r = numpy.zeros((num_filters, pos_right + pos_left))

		meme = open('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/tomtom_meme.txt', 'w')
		meme.write('MEME version 4\n')
		meme.write('\n')
		meme.write('ALPHABET= ACGT')
		meme.write('\n')
		meme.write('strands: + -')
		meme.write('\n')

		filter_width = 8

		for k in range(0, num_filters) :


			filter_activations = numpy.concatenate([get_layer0_activations_k(i, k) for i in xrange(n_batches)], axis=0).reshape((y_test_hat.shape[0], activation_length))
			filter_activations = filter_activations[y_test_hat_isnan == False, :]

			valid_activations = numpy.zeros(filter_activations.shape)
			#valid_activations[L_index == 20, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]))), (len(numpy.nonzero(L_index == 20)[0]), activation_length))
			valid_activations[L_index == 22, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]))), (len(numpy.nonzero(L_index == 22)[0]), activation_length))
			filter_activations = numpy.multiply(filter_activations, valid_activations)

			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))
			pos_activation = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :]
			filter_L = L_index[spike_index]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-5000:]#5000

			PFM = numpy.zeros((filter_width, self.num_features))
			for ii in range(0, len(top_scoring_index)) :
				i = top_scoring_index[ii]

				filter_input = numpy.asarray(filter_inputs[i, :].todense()).reshape((self.input_size, self.num_features))[0:self.left_random_size, :]
				#filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :]
				filter_input = filter_input[max_spike[i]:max_spike[i]+filter_width, :] * filter_activations[i, max_spike[i]]

				PFM = PFM + filter_input

			print('Motif ' + str(k))

			#Estimate PPM motif properties
			PPM = numpy.zeros(PFM.shape)
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PPM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])

			meme.write('MOTIF Layer_1_Filter_' + str(k) + '\n')
			meme.write('letter-probability matrix: alength= 4 w= 8 nsites= 5000\n')
			for i in range(0, PPM.shape[0]) :
				for j in range(0, 4) :
					meme.write(' ' + str(round(PPM[i, j], 6)) + ' ')
				meme.write('\n')
			meme.write('\n')


			'''logodds_test_hat_aligned = []#numpy.zeros(y_test_hat.shape[0] * y_test_hat_c.shape[1])
			pos_activation_aligned = []#numpy.zeros((y_test_hat.shape[0] * y_test_hat_c.shape[1], pos_right + pos_left))

			j = 0
			for cut_pos in range(cut_start, cut_end + 1) :
				y_test_hat_c = numpy.ravel(y_test_hat[:, j])
				logodds_test_hat_c = safe_log(y_test_hat_c / (1 - y_test_hat_c))
				logodds_test_hat_c_isinf = numpy.isinf(logodds_test_hat_c)

				logodds_test_hat_c = logodds_test_hat_c[logodds_test_hat_c_isinf == False]
				pos_activation_c = pos_activation[logodds_test_hat_c_isinf == False, :]

				logodds_test_hat_aligned.append(logodds_test_hat_c[:])
				#logodds_test_hat_aligned[j * y_test_hat_c.shape[0] : (j + 1) * y_test_hat_c.shape[0]] = logodds_test_hat_c[:]

				pos_activation_c = pos_activation_c[:, cut_start - pos_left + j: cut_start + pos_right + j]

				pos_activation_aligned.append(pos_activation_c[:, :])
				#pos_activation_aligned[cut_start - pos_left + j: cut_start + pos_right + j, :] = pos_activation_c[:, :]

				j += 1

			logodds_test_hat_aligned = numpy.concatenate(logodds_test_hat_aligned, axis=0)
			pos_activation_aligned = numpy.concatenate(pos_activation_aligned, axis=0)

			logodds_test_hat_aligned_avg = numpy.average(logodds_test_hat_aligned)
			logodds_test_hat_aligned_std = numpy.sqrt(numpy.dot(logodds_test_hat_aligned - logodds_test_hat_aligned_avg, logodds_test_hat_aligned - logodds_test_hat_aligned_avg))

			for pos in range(0, pos_activation_aligned.shape[1]) :

				pos_activation_k_pos = numpy.ravel(pos_activation_aligned[:, pos])
				pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
				pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

				cov_pos = numpy.dot(logodds_test_hat_aligned - logodds_test_hat_aligned_avg, pos_activation_k_pos - pos_activation_k_pos_avg)
				r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_hat_aligned_std)

				if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
					pos_r[k, pos] = r_k_pos

				j += 1'''

			logo_name = "avg_motif_" + str(k) + ".png"

			logotitle = "Layer 1 Filter " + str(k)
			self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + logo_name, 8, logotitle=logotitle, logo_file_type='png')

			logo_name = "avg_motif_" + str(k) + ".svg"

			logotitle = "Layer 1 Filter " + str(k)
			self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + logo_name, 8, logotitle=logotitle, logo_file_type='svg')

		meme.close()

		#All-filter positional Pearson r
		'''f = plt.figure(figsize=(8, 16))

		avg_r_sort_index = numpy.argsort(numpy.ravel(numpy.mean(pos_r, axis=1)))

		pos_r = pos_r[avg_r_sort_index, :]

		plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Cutsite Pearson r heatmap')
		
		plt.xticks([0, pos_left, pos_left + pos_right], [-pos_left, 0, pos_right])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, avg_r_sort_index)

		plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "r_pos_cutcenter_" + str(pos_left) + "up" + str(pos_right) + "down.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/deconv/avg_filter_cuts/' + "r_pos_cutcenter_" + str(pos_left) + "up" + str(pos_right) + "down.svg")
		plt.close()'''


	def get_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_global_onesided2_cuts_dropout/', seq_length=6, normalize=False, logotitle="", logo_file_type='png') :

		if normalize == True :
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
				#PFM[i, :] *= 10000.0
			#print(PFM)

		#Create weblogo from API
		logo_output_format = logo_file_type#"png"
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
		                                 #resolution = 400,
		                                 logo_start=1, logo_end=seq_length)

		#Create logo
		logo_format = weblogolib.LogoFormat(data, options)

		#Generate image
		formatter = weblogolib.formatters[logo_output_format]
		png = formatter(data, logo_format)

		#Write it
		with open(file_path, "w") as f:
		    f.write(png)


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

			plt.savefig("cnn_motif_analysis/fullseq_global/kernal/kernel" + str(k) + ".png")
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


	def __init__(self, train_set, valid_set, learning_rate=0.1, drop=0, n_epochs=30, nkerns=[30, 40, 50], batch_size=50, num_features=4, randomized_regions=[(2, 37), (45, 80)], load_model=True, train_model_flag=False, store_model=False, dataset='default', store_as_dataset='default', cell_line='default'):
		numpy.random.seed(23455)
		rng = numpy.random.RandomState(23455)

		srng = RandomStreams(rng.randint(999999))

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
			rng,
			input=layer0_left.output,
			deactivated_filter=None,
			deactivated_output=None,
			image_shape=(batch_size, nkerns[0], 89, 1),
			filter_shape=(nkerns[1], nkerns[0], 6, 1),
			poolsize=(1, 1),
			activation_fn=modded_relu#relu
			,load_model = load_model,
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
			#srng,
			input=layer3_input,
			n_in=nkerns[1] * (84) * 1 + 1,#n_in=nkerns[1] * (169) * 1 + 1,
			n_out=200,
			activation=modded_relu#relu#T.tanh#relu#T.tanh
			,load_model = load_model,
			w_file='model_store/' + dataset + '_' + cell_line + '_mlp_w',
			b_file='model_store/' + dataset + '_' + cell_line + '_mlp_b',
			store_as_w_file='model_store/' + store_as_dataset + '_' + cell_line + '_mlp_w',
			store_as_b_file='model_store/' + store_as_dataset + '_' + cell_line + '_mlp_b'
		)

		layer3_output = layer3.output

		'''if drop != 0 and train_model_flag == True :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 1)
		elif drop != 0 :
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = 0)'''

		train_drop = T.lscalar()
		self.train_drop = train_drop
		if drop != 0 :
			print('Using dropout = ' + str(drop))
			layer3_output = self.dropout_layer(srng, layer3.output, drop, train = train_drop)

		layer4_input = T.concatenate([layer3_output, L_input], axis=1)
		#layer4_input = layer3.output

		# classify the values of the fully-connected sigmoidal layer
		layer4 = LogisticRegression(input=layer4_input, L_input=L_input, n_in=200 + 36, n_out=self.input_size + 1, load_model = load_model,
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
		cost = layer4.negative_log_likelihood(y, L_input)

		# create a function to compute the mistakes that are made by the model
		validate_model = theano.function(
			[index],
			layer4.log_loss(y, L_input),
			givens={
				x_left: self.reshape_batch(valid_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
				x_right: self.reshape_batch(valid_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[1][0]:randomized_regions[1][1]],
				y: Tsparse.basic.dense_from_sparse(valid_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
				L_input: valid_set_L[index * batch_size: (index + 1) * batch_size, :],
				d_input: valid_set_d[index * batch_size: (index + 1) * batch_size, :]
				,train_drop: 0
			},
			on_unused_input='ignore'
		)

		validate_rsquare = theano.function(
			[index],
			layer4.rsquare(y, L_input),
			givens={
				x_left: self.reshape_batch(valid_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(valid_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: Tsparse.basic.dense_from_sparse(valid_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
				y: Tsparse.basic.dense_from_sparse(valid_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
				y: Tsparse.basic.dense_from_sparse(valid_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
				L_input: valid_set_L[index * batch_size: (index + 1) * batch_size, :]
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
			(param_i, param_i - learning_rate * grad_i)
			for param_i, grad_i in zip(params, grads)
		]

		train_model = theano.function(
			[index],
			cost,
			updates=updates,
			givens={
				x_left: self.reshape_batch(train_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),
				x_right: self.reshape_batch(train_set_x, index, randomized_regions[1][0], randomized_regions[1][1]),
				y: Tsparse.basic.dense_from_sparse(train_set_y[index * batch_size: (index + 1) * batch_size]).astype(theano.config.floatX),
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
					cost_ij = train_model(minibatch_index)
					
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

						# if we got the best validation score until now
						if this_validation_loss < best_validation_loss:

							#improve patience if loss improvement is good enough
							if this_validation_loss < best_validation_loss *  \
							   improvement_threshold:
								patience = max(patience, iter * patience_increase)

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


def evaluate_cnn(dataset='general_cuts_antimisprime_orig'):#_pasaligned

	count_filter=0#5#0

	input_datasets = load_input_data(dataset, shuffle=True, shuffle_all=False, count_filter=count_filter, balance_all_libraries=True)
	
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

	output_datasets = load_output_data(dataset, shuffle_index, shuffle_all_index, count_filter=count_filter, balance_all_libraries=True)
	
	train_set_y = output_datasets[0]
	valid_set_y = output_datasets[1]
	test_set_y = output_datasets[2]


	#run_name = '_Global2_Onesided2Cuts_DoubleDope_Simple_TOMM5_APA_Six_30_31_34'
	run_name = '_Global2_Onesided2Cuts2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34'
	#run_name = '_Global2_Onesided2Cuts2AntimisprimeOrigDropout_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_pasaligned'


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



	batch_size = 1#50
	
	cnn = DualCNN(
		(train_set_x, train_set_y, train_set_L, train_set_d),
		(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
		learning_rate=0.1,
		drop=0.2,
		n_epochs=10,
		nkerns=[70, 110, 70],
		#nkerns=[128, 256, 70],#_medium
		batch_size=batch_size,
		num_features=4,
		randomized_regions=[(0, 185), (185, 185)],
		load_model=True,
		train_model_flag=False,
		store_model=False,
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts_finetuned_TOMM5_APA_Six_30_31_34'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2_finetuned_TOMM5_APA_Six_30_31_34'
		
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34'
		dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned',
		store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_nopool',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_nopool'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_medium',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_medium'
	)
	
	print('Trained sublib bias terms:')
	lrW = cnn.output_layer.W.eval()
	lrW = numpy.ravel(lrW[lrW.shape[0] - 36:, 1])
	for i in range(0, len(lrW)) :
		if lrW[i] != 0 :
			print(str(i) + ": " + str(lrW[i]))

	#cnn.set_data(test_set_x, test_set_y, test_set_L)
	#cnn.set_saliency_functions((test_set_x, test_set_y, test_set_L))

	#cnn.generate_heat_maps()
	#cnn.generate_local_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='pos_and_neg')
	#cnn.generate_global_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='pos')
	#cnn.generate_global_saliency_sequence_logos((test_set_x, test_set_y, test_set_L), pos_neg='neg')
	#cnn.generate_local_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='pos_and_neg')
	#cnn.generate_global_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='pos')
	#cnn.generate_global_saliency_sequence_logos_level2((test_set_x, test_set_y, test_set_L), pos_neg='neg')
	#cnn.generate_sequence_logos((test_set_x, test_set_y, test_set_L, test_set_d))
	#cnn.generate_avgcut_heatmap((test_set_x, test_set_y, test_set_L, test_set_d))

	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)
	test_fold_cuts(cnn)


	'''cnn.generate_sequence_logos_cuts((test_set_x, test_set_y, test_set_L, test_set_d), 25, 25)
	cnn.generate_sequence_logos_cuts((test_set_x, test_set_y, test_set_L, test_set_d), 20, 20)
	cnn.generate_sequence_logos_cuts((test_set_x, test_set_y, test_set_L, test_set_d), 60, 20)'''

	'''cut_start = 72#65#80
	cut_end = 77#70#85
	#cnn.generate_sequence_logos_cuts((test_set_x, test_set_y, test_set_L, test_set_d), 25, 25)
	cnn.generate_motif_cut_heatmap((test_set_x, test_set_y, test_set_L, test_set_d))'''

	#cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 82, 83, 'longcut_2pos')
	#cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 71, 72, 'midcut_2pos')
	#cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 65, 66, 'shortcut_2pos')

	#cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 72, 73, 'midcut')
	#cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 65, 66, 'shortcut')'''

	'''cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 87, 87, 'longlongcut_1pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 61, 61, 'shortshortcut_1pos')

	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 87, 88, 'longlongcut_2pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 61, 62, 'shortshortcut_2pos')'''


	'''cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 82, 82, 'longcut_1pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 71, 71, 'midcut_1pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 65, 65, 'shortcut_1pos')

	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 82, 83, 'longcut_2pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 71, 72, 'midcut_2pos')
	cnn.generate_sequence_logos_long_cut((test_set_x, test_set_y, test_set_L, test_set_d), 65, 66, 'shortcut_2pos')'''

	'''cut_start = 72#65#80
	cut_end = 77#70#85
	cnn.generate_sequence_logos_long_cut_level2((test_set_x, test_set_y, test_set_L, test_set_d), 82, 83, 'longcut')
	cnn.generate_sequence_logos_long_cut_level2((test_set_x, test_set_y, test_set_L, test_set_d), 65, 66, 'shortcut')
	cnn.generate_sequence_logos_long_cut_level2((test_set_x, test_set_y, test_set_L, test_set_d), 72, 73, 'midcut')'''


	#get_global_saliency(cnn, (test_set_x, test_set_y, test_set_L))

	#debug_libraries(cnn, test_set_x, test_set_y, test_set_L)
	#print(1 + '')

	#store_predictions(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name)
	#cross_test(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name)
	print(1 + '')

def test_fold_cuts(cnn) :

	data = pd.read_csv('apa_general_cuts_antimisprime_dse_folded_cuttests.csv',sep=',')
	print(len(data))

	seqs = list(data.seq.str.slice(1,186).values)
	ys = numpy.ravel(data.observed_logodds.values)
	mfes = numpy.ravel(data.mfe.values)
	cut_poses = numpy.ravel(data.cut_pos.values)

	y_hats = []

	for i in range(0, len(seqs)) :

		if i % 10000 == 0 :
			print('Predicting seq ' + str(i))

		seq = seqs[i]
		y = ys[i]
		mfe = mfes[i]
		test_cut_index = int(cut_poses[i])

		x = numpy.zeros((1, 185, 4))
		for j in range(0, 185) :
			if seq[j] == 'A' :
				x[0, j, 0] = 1
			elif seq[j] == 'C' :
				x[0, j, 1] = 1
			elif seq[j] == 'G' :
				x[0, j, 2] = 1
			elif seq[j] == 'T' :
				x[0, j, 3] = 1

		L_fake = numpy.zeros((1, 36))
		L_fake[0, 22] = 1

		d_fake = numpy.zeros((1, 1))
		positions = numpy.arange(186).tolist()

		cuts_hat = numpy.ravel(cnn.get_online_prediction_distrib(x, L_fake, d_fake, positions))

		prox_cuts_hat = cuts_hat[60:100]
		prox_norm = numpy.sum(prox_cuts_hat)
		prox_cuts_hat[:] = prox_cuts_hat[:] / prox_norm

		
		test_cut = numpy.sum(prox_cuts_hat[test_cut_index-1:test_cut_index+1])

		y_hat = numpy.log(test_cut / (1.0 - test_cut))
		y_hats.append(y_hat)

	data['predicted_logodds'] = y_hats

	data = data[['seq', 'mfe', 'observed_logodds', 'predicted_logodds', 'cut_pos']]

	data.to_csv('apa_general_cuts_antimisprime_dse_folded_cuttests_pred.csv', header=True, index=False, sep=',')




def store_predictions(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name) :
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
	}


	y_test_hat = numpy.ravel(numpy.array(cnn.get_prediction()))
	y_test = numpy.ravel(numpy.array(cnn.get_target()))
	L_test = numpy.array(test_set_L.eval())[:y_test_hat.shape[0],:]
	X_test = numpy.array(test_set_x.eval().todense())[:y_test_hat.shape[0], :]
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1] / 4, 4))

	logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
	logodds_test = safe_log(y_test / (1 - y_test))
	logodds_test_isinf = numpy.isinf(logodds_test)
	logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
	logodds_test = logodds_test[logodds_test_isinf == False]
	y_test_hat = y_test_hat[logodds_test_isinf == False]
	y_test = y_test[logodds_test_isinf == False]
	X_test = X_test[logodds_test_isinf == False, :]

	L_test = L_test[logodds_test_isinf == False,:]
	L_test_sum = numpy.ravel(numpy.sum(L_test, axis=0))
	L_test = numpy.ravel(numpy.argmax(L_test, axis=1))


	with open('test_predictions_' + run_name + '.csv', 'w') as f :
		f.write('seq\tlibrary\tlibrary_name\tobserved_logodds\tpredicted_logodds\n')

		for i in range(0, X_test.shape[0]) :
			seq = translate_matrix_to_seq(X_test[i, :, :])

			li = L_test[i]
			lib_name = lib_dict[li]

			f.write(seq + '\t' + str(int(li)) + '\t' + lib_name + '\t' + str(round(logodds_test[i], 3)) + '\t' + str(round(logodds_test_hat[i], 3)) + '\n')

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
	y_test = cnn.get_target()
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

def cross_test(cnn, test_set_x, test_set_y, test_set_L, test_set_d, run_name) :
	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)

	lib_map = [
		[2, 'TOMM5 2', 'red', [0, 0]],
		[5, 'TOMM5 5', 'red', [0, 1]],
		[8, 'TOMM5 8', 'red', [0, 2]],
		[11, 'TOMM5 11', 'red', [0, 3]],
		[20, 'Double Dope', 'blue', [1, 0]],
		[22, 'Simple', 'green', [2, 0]],
		[30, 'Six 30', 'purple', [3, 0]],
		[31, 'Six 31', 'purple', [3, 1]],
		[32, 'Six 32', 'purple', [3, 2]],
		[33, 'Six 33', 'purple', [3, 3]],
		[34, 'Six 34', 'purple', [3, 4]],
		[35, 'Six 35', 'purple', [3, 5]],
	]


	y_test_hat = cnn.get_prediction()
	y_test = cnn.get_target()#test_set_y.eval()[:y_test_hat.shape[0],1]
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

	for plot_type in ['ratios', 'logodds'] :

		f, axarr = plt.subplots(5, 6, figsize=(30, 20), sharex=False)#, sharey=True

		for i in range(0, len(lib_map)) :
			lib = lib_map[i]
			lib_index = lib[0]
			lib_name = lib[1]
			lib_color = lib[2]
			lib_grid = lib[3]

			if L_test_sum[lib_index] <= 0 :
				continue

			y_test_hat_curr = y_test_hat[L_test == lib_index]
			y_test_curr = y_test[L_test == lib_index]
			logodds_test_hat_curr = logodds_test_hat[L_test == lib_index]
			logodds_test_curr = logodds_test[L_test == lib_index]


			SSE_test = (logodds_test_curr - logodds_test_hat_curr).T.dot(logodds_test_curr - logodds_test_hat_curr)
			logodds_test_average = numpy.average(logodds_test_curr, axis=0)
			SStot_test = (logodds_test_curr - logodds_test_average).T.dot(logodds_test_curr - logodds_test_average)
			logodds_test_rsquare = 1.0 - (SSE_test / SStot_test)

			logodds_test_curr_norm = logodds_test_curr - numpy.mean(logodds_test_curr)
			logodds_test_hat_curr_norm = logodds_test_hat_curr - numpy.mean(logodds_test_hat_curr)
			SSE_test = (logodds_test_curr_norm - logodds_test_hat_curr_norm).T.dot(logodds_test_curr_norm - logodds_test_hat_curr_norm)
			logodds_test_average = numpy.average(logodds_test_curr_norm, axis=0)
			SStot_test = (logodds_test_curr_norm - logodds_test_average).T.dot(logodds_test_curr_norm - logodds_test_average)
			logodds_test_rsquare_norm = 1.0 - (SSE_test / SStot_test)

			#Calculate Pearson r
			logodds_test_hat_avg = numpy.average(logodds_test_hat_curr)
			logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_hat_curr - logodds_test_hat_avg))

			logodds_test_avg = numpy.average(logodds_test_curr)
			logodds_test_std = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg, logodds_test_curr - logodds_test_avg))

			cov = numpy.dot(logodds_test_hat_curr - logodds_test_hat_avg, logodds_test_curr - logodds_test_avg)
			test_r = cov / (logodds_test_hat_std * logodds_test_std)

			logodds_test_rsquare_norm = test_r * test_r

			if plot_type == 'ratios' :
				axarr[lib_grid[0], lib_grid[1]].scatter(y_test_hat_curr, y_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color=lib_color)
				axarr[lib_grid[0], lib_grid[1]].plot([0,1], [0,1], '-y')
			elif plot_type == 'logodds' :
				axarr[lib_grid[0], lib_grid[1]].scatter(logodds_test_hat_curr, logodds_test_curr, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color=lib_color)
			

			plt.sca(axarr[lib_grid[0], lib_grid[1]])
			if plot_type == 'ratios' :
				plt.axis([0, 1, 0, 1])
			elif plot_type == 'logodds' :
				plt.axis([numpy.min(logodds_test_hat_curr), numpy.max(logodds_test_hat_curr), numpy.min(logodds_test_curr), numpy.max(logodds_test_curr)])
			plt.title('R^2 = ' + str(round(logodds_test_rsquare, 2)) + ', r = ' + str(round(test_r, 2)) + ', nR^2 = ' + str(round(logodds_test_rsquare_norm, 2)))
			if lib_grid[1] == 0 :
				plt.ylabel(lib_name)
		

		SSE_test = (logodds_test - logodds_test_hat).T.dot(logodds_test - logodds_test_hat)
		logodds_test_average = numpy.average(logodds_test, axis=0)
		SStot_test = (logodds_test - logodds_test_average).T.dot(logodds_test - logodds_test_average)
		logodds_test_rsquare = 1.0 - (SSE_test / SStot_test)

		logodds_test_norm = logodds_test - numpy.mean(logodds_test)
		logodds_test_hat_norm = logodds_test_hat - numpy.mean(logodds_test_hat)
		SSE_test = (logodds_test_norm - logodds_test_hat_norm).T.dot(logodds_test_norm - logodds_test_hat_norm)
		logodds_test_average = numpy.average(logodds_test_norm, axis=0)
		SStot_test = (logodds_test_norm - logodds_test_average).T.dot(logodds_test_norm - logodds_test_average)
		logodds_test_rsquare_norm = 1.0 - (SSE_test / SStot_test)

		#Calculate Pearson r
		logodds_test_hat_avg = numpy.average(logodds_test_hat)
		logodds_test_hat_std = numpy.sqrt(numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test_hat - logodds_test_hat_avg))

		logodds_test_avg = numpy.average(logodds_test)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

		cov = numpy.dot(logodds_test_hat - logodds_test_hat_avg, logodds_test - logodds_test_avg)
		test_r = cov / (logodds_test_hat_std * logodds_test_std)

		logodds_test_rsquare_norm = test_r * test_r


		if plot_type == 'ratios' :
			axarr[4, 0].scatter(y_test_hat, y_test, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='black')
			axarr[4, 0].plot([0,1], [0,1], '-y')
		elif plot_type == 'logodds' :
			axarr[4, 0].scatter(logodds_test_hat, logodds_test, s = numpy.pi * (2 * numpy.ones(1))**2, alpha=0.05, color='black')

		plt.sca(axarr[4, 0])
		if plot_type == 'ratios' :
			plt.axis([0, 1, 0, 1])
		elif plot_type == 'logodds' :
			plt.axis([numpy.min(logodds_test_hat), numpy.max(logodds_test_hat), numpy.min(logodds_test), numpy.max(logodds_test)])
		plt.title('R^2 = ' + str(round(logodds_test_rsquare, 2)) + ', r = ' + str(round(test_r, 2)) + ', nR^2 = ' + str(round(logodds_test_rsquare_norm, 2)))
		plt.ylabel('Total')


		f.tight_layout()
		
		plt.subplots_adjust(top=0.93, wspace = 0.6)
		f.suptitle('APA CNN Model tested on separate libraries (Predicted vs. Observed)', fontsize=16)
		
		plt.savefig("cross_test" + run_name + "_" + plot_type + ".png")
		#plt.show()
		plt.close()

def safe_log(x, minval=0.02):
    return numpy.log(x.clip(min=minval))

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
		if X_point[j, 0] == 1 :
			seq += "A"
		elif X_point[j, 1] == 1 :
			seq += "C"
		elif X_point[j, 2] == 1 :
			seq += "G"
		elif X_point[j, 3] == 1 :
			seq += "T"
		else :
			seq += "."
	return seq

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
			seq += "."
	return seq

if __name__ == '__main__':
	evaluate_cnn('general_cuts_antimisprime_orig')#_pasaligned
