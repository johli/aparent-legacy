
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

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import theano.sparse as Tsparse

from logistic_sgd_global_onesided2_cuts_cenpb1 import LogisticRegression, load_input_data, load_output_data
from mlp import HiddenLayer

#import pylab as pl
#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

import matplotlib as mpl
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec

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

class CutCNN(object):

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

	def get_class_score(self, i=-1):
		if i == -1:
			return numpy.concatenate([self.class_score(i) for i in xrange(self.n_batches)])
		else:
			return self.class_score(i)

	def get_online_prediction(self, data_x, data_L, data_d):
		return self.online_predict(data_x, data_L, data_d)

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

	def generate_sequence_logos_level2(self, test_set):
			test_set_x, test_set_y, test_set_L = test_set
			self.set_data(test_set_x, test_set_y, test_set_L)

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

			get_layer1_activations = theano.function(
				[index],
				layer1.activation,
				givens={
					x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
				},
				on_unused_input='ignore'
			)

			activations = numpy.concatenate([get_layer1_activations(i) for i in xrange(n_batches)], axis=0)

			print('Computed activations')
			
			input_x = input_x[:activations.shape[0],:]
			L_index = L_index[:activations.shape[0]]
			input_x = numpy.asarray(input_x.todense()).reshape((activations.shape[0], self.input_size, self.num_features))[:, 0:self.left_random_size, :]

			y_test_hat = self.get_prediction()
			y_test = test_set_y.eval()[:y_test_hat.shape[0],1]
			logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
			logodds_test = safe_log(y_test / (1 - y_test))

			logodds_test_isinf = numpy.isinf(logodds_test)
			logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
			logodds_test = logodds_test[logodds_test_isinf == False]
			activations = activations[logodds_test_isinf == False, :, :, :]
			input_x = input_x[logodds_test_isinf == False, :, :]
			L_index = L_index[logodds_test_isinf == False]

			print('Reshaped and filtered activations')

			#logodds_test_hat injection
			logodds_test = logodds_test_hat

			logodds_test_avg = numpy.average(logodds_test)
			logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

			max_activation = numpy.zeros((activations.shape[1], activations.shape[0]))
			max_activation_up = numpy.zeros((activations.shape[1], activations.shape[0]))
			max_activation_pas = numpy.zeros((activations.shape[1], activations.shape[0]))
			max_activation_dn = numpy.zeros((activations.shape[1], activations.shape[0]))
			pos_activation = numpy.zeros((activations.shape[1], activations.shape[0], activations.shape[2]))

			pos_r = numpy.zeros((activations.shape[1], activations.shape[2]))

			filter_width = 19


			valid_activations = numpy.zeros(activations.shape)
		
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


			'''libvar_2 = ('X' * (80 - 1)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('X' * 20) + ('X' * (83 - 7))
			libvar_2_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_2[i] == 'V' :
					libvar_2_id[i] = 1
			libvar_2_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_2_id[2 * i] == 1 and libvar_2_id[2 * i + 1] == 1 :
					libvar_2_idd[i] = 1
			libvar_2_id = libvar_2_idd

			libvar_8 = ('X' * (80 - 1)) + ('V' * (20 - 7)) + ('X' * (20 + 7)) + ('X' * 33) + ('V' * (20 - 7)) + ('X' * (83 - 7 + 7))
			libvar_8_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_8[i] == 'V' :
					libvar_8_id[i] = 1
			libvar_8_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_8_id[2 * i] == 1 and libvar_8_id[2 * i + 1] == 1 :
					libvar_8_idd[i] = 1
			libvar_8_id = libvar_8_idd

			libvar_5 = ('X' * (80 - 1)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('X' * 20) + ('X' * (83 - 7))
			libvar_5_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_5[i] == 'V' :
					libvar_5_id[i] = 1
			libvar_5_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_5_id[2 * i] == 1 and libvar_5_id[2 * i + 1] == 1 :
					libvar_5_idd[i] = 1
			libvar_5_id = libvar_5_idd

			libvar_11 = ('X' * (80 - 1)) + ('X' * 20) + ('V' * (20 - 7)) + ('X' * (33 + 7)) + ('V' * (20 - 7)) + ('X' * (83 - 7 + 7))
			libvar_11_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_11[i] == 'V' :
					libvar_11_id[i] = 1
			libvar_11_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_11_id[2 * i] == 1 and libvar_11_id[2 * i + 1] == 1 :
					libvar_11_idd[i] = 1
			libvar_11_id = libvar_11_idd

			#APA_SYM_PRX
			libvar_20 = ('X' * (100 - 1)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7))
			libvar_20_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_20[i] == 'V' :
					libvar_20_id[i] = 1
			libvar_20_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_20_id[2 * i] == 1 and libvar_20_id[2 * i + 1] == 1 :
					libvar_20_idd[i] = 1
			libvar_20_id = libvar_20_idd

			libvar_21 = ('X' * (15 - 1)) + ('V' * (71 - 7)) + ('X' * (14 + 7)) + ('V' * (71 - 7)) + ('X' * (85 - 7 + 7))
			libvar_21_id = numpy.zeros(255 - 7)
			for i in range(0, 255 - 7) :
				if libvar_21[i] == 'V' :
					libvar_21_id[i] = 1
			libvar_21_idd = numpy.zeros(120)
			for i in range(0, 120) :
				if libvar_21_id[2 * i] == 1 and libvar_21_id[2 * i + 1] == 1 :
					libvar_21_idd[i] = 1
			libvar_21_id = libvar_21_idd'''


			pos_to_libs = [
				#[libvar_2_id, 2],
				#[libvar_8_id, 8],
				#[libvar_5_id, 5],
				#[libvar_11_id, 11],
				[libvar_20_id, 20],
				#[libvar_22_id, 22],
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


			#valid_activations[L_index == 2, :, :, :] = numpy.reshape(numpy.tile(libvar_2_id, (len(numpy.nonzero(L_index == 2)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 2)[0]), activations.shape[1], activations.shape[2], 1))
			#valid_activations[L_index == 8, :, :, :] = numpy.reshape(numpy.tile(libvar_8_id, (len(numpy.nonzero(L_index == 8)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 8)[0]), activations.shape[1], activations.shape[2], 1))
			#valid_activations[L_index == 5, :, :, :] = numpy.reshape(numpy.tile(libvar_5_id, (len(numpy.nonzero(L_index == 5)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 5)[0]), activations.shape[1], activations.shape[2], 1))
			#valid_activations[L_index == 11, :, :, :] = numpy.reshape(numpy.tile(libvar_11_id, (len(numpy.nonzero(L_index == 11)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 11)[0]), activations.shape[1], activations.shape[2], 1))
			valid_activations[L_index == 20, :, :, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 20)[0]), activations.shape[1], activations.shape[2], 1))
			#valid_activations[L_index == 22, :, :, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 22)[0]), activations.shape[1], activations.shape[2], 1))

			
			activations = numpy.multiply(activations, valid_activations)

			k_r = numpy.zeros(activations.shape[1])

			#(num_data_points, num_filters, seq_length, 1)
			for k in range(0, activations.shape[1]) :
				filter_activations = activations[:, k, :, :].reshape((activations.shape[0], activations.shape[2]))
				total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

				max_activation[k, :] = numpy.ravel(numpy.max(filter_activations, axis=1))

				#Region-specific max activations
				max_activation_up[k, :] = numpy.ravel(numpy.max(filter_activations[:,:22], axis=1))
				max_activation_pas[k, :] = numpy.ravel(numpy.max(filter_activations[:,22:25], axis=1))
				max_activation_dn[k, :] = numpy.ravel(numpy.max(filter_activations[:,25:47], axis=1))
				max_activation_comp = numpy.ravel(numpy.max(filter_activations[:,47:], axis=1))

				pos_activation[k, :, :] = filter_activations[:, :]
				
				spike_index = numpy.nonzero(total_activations > 0)[0]

				filter_activations = filter_activations[spike_index, :]

				filter_inputs = input_x[spike_index, :, :]
				filter_L = L_index[spike_index]

				max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

				top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
				top_scoring_index = top_scoring_index[len(top_scoring_index)-500:]

				PFM = numpy.zeros((filter_width, self.num_features))
				for ii in range(0, len(top_scoring_index)) :
					i = top_scoring_index[ii]

					'''input_saliency_from_conv1 = self.get_input_conv1_saliency(spike_index[i], k, max_spike[i])
					input_saliency_from_conv1_index = input_saliency_from_conv1 > 0
					input_saliency_from_conv1_id = numpy.zeros(input_saliency_from_conv1.shape)
					input_saliency_from_conv1_id[input_saliency_from_conv1_index] = 1
					input_saliency_from_conv1_id = input_saliency_from_conv1_id[0, 2*max_spike[i]:2*max_spike[i]+filter_width, :]'''
					#filter_input = numpy.multiply(filter_inputs[i, 2*max_spike[i]:2*max_spike[i]+filter_width, :], input_saliency_from_conv1_id) #* filter_activations[i, max_spike[i]]
					filter_input = filter_inputs[i, 2*max_spike[i]:2*max_spike[i]+filter_width, :]

					PFM = PFM + filter_input

				#print(k)
				#print(PFM)

				'''#Calculate Pearson r
				logodds_test_curr = logodds_test
				logodds_test_avg_curr = logodds_test_avg
				logodds_test_std_curr = logodds_test_std
				max_activation_k = numpy.ravel(max_activation[k, :])

				max_activation_k = max_activation_k[L_index > 5]
				logodds_test_curr = logodds_test[L_index > 5]

				max_activation_k_avg = numpy.average(max_activation_k)
				max_activation_k_std = numpy.sqrt(numpy.dot(max_activation_k - max_activation_k_avg, max_activation_k - max_activation_k_avg))

				logodds_test_avg_curr = numpy.average(logodds_test_curr)
				logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

				cov = numpy.dot(logodds_test_curr - logodds_test_avg_curr, max_activation_k - max_activation_k_avg)
				r = cov / (max_activation_k_std * logodds_test_std_curr)
				print('r = ' + str(round(r, 2)))'''

				print('(Layer 2) Motif ' + str(k))

				#Calculate Pearson r
				logodds_test_curr = logodds_test
				logodds_test_avg_curr = logodds_test_avg
				logodds_test_std_curr = logodds_test_std

				max_activation_regions = [
					[numpy.ravel(max_activation[k, :]), 'All region'],
					#[numpy.ravel(max_activation_up[k, :]), 'Upstream'],
					#[numpy.ravel(max_activation_pas[k, :]), 'PAS'],
					#[numpy.ravel(max_activation_dn[k, :]), 'Downstream'],
					#[numpy.ravel(max_activation_comp), 'Distal']
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

					if region[1] == 'All region' :
						k_r[k] = r
					elif region[1] == 'Upstream' :
						r_up_k = r
					elif region[1] == 'PAS' :
						r_pas_k = r
					elif region[1] == 'Downstream' :
						r_dn_k = r

				print('')

				prev_selection_libs_str = 'X'
				for pos in range(0, activations.shape[2]) :

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
						
						pos_activation_curr = pos_activation[:, whitelist_index, :]
						logodds_test_curr = logodds_test[whitelist_index]
					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					if curr_selection_libs_str == '' :
						continue

					pos_activation_k_pos = numpy.ravel(pos_activation_curr[k, :, pos])
					pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
					pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

					cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k_pos - pos_activation_k_pos_avg)
					r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_std_curr)

					if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
						pos_r[k, pos] = r_k_pos

					prev_selection_libs_str = curr_selection_libs_str
					pos_activation_prev = pos_activation_curr
					logodds_test_prev = logodds_test_curr

				logo_name = "avg_motif_" + str(k) + ".png"
				logo_name_normed = "avg_motif_" + str(k) + '_normed' + ".png"

				logotitle = "Layer 2 Filter " + str(k)
				self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter_level2/' + logo_name, 19, logotitle=logotitle)
				#self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter_level2/' + logo_name_normed, 19, normalize=True, score=r)

			#All-filter positional Pearson r
			f = plt.figure(figsize=(18, 16))

			plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
			plt.colorbar()

			plt.xlabel('Sequence position')
			plt.title('Prox. selection Pearson r for all layer 2 filters')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.xticks([0, 12, 24, 36, 48, 60, 72, 84], [0 - 24, 12 - 24, 24 - 24, 36 - 24, 48 - 24, 60 - 24, 72 - 24, 84 - 24])
			plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, numpy.arange(pos_r.shape[0]))#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

			plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

			plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter_level2/' + "r_pos.png")
			plt.close()

			f = plt.figure(figsize=(18, 16))

			plt.pcolor(numpy.repeat(pos_r, 2, axis=1),cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
			plt.colorbar()

			plt.xlabel('Sequence position')
			plt.title('Prox. selection Pearson r for all layer 2 filters')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.xticks([0*2, 12*2, 24*2, 36*2, 48*2, 60*2, 72*2, 84*2], [0*2 - 24*2, 12*2 - 24*2, 24*2 - 24*2, 36*2 - 24*2, 48*2 - 24*2, 60*2 - 24*2, 72*2 - 24*2, 84*2 - 24*2])
			plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, numpy.arange(pos_r.shape[0]))#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

			plt.axis([0, pos_r.shape[1] * 2, 0, pos_r.shape[0]])

			plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter_level2/' + "r_pos_projected.png")
			plt.close()
			

	def generate_sequence_logos(self, test_set):
		test_set_x, test_set_y, test_set_L = test_set
		self.set_data(test_set_x, test_set_y, test_set_L)

		layer0_left = self.layer0_left

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

		get_layer0_activations = theano.function(
			[index],
			layer0_left.activation,
			givens={
				x_left: self.reshape_batch(test_set_x, index, randomized_regions[0][0], randomized_regions[0][1]),#Tsparse.basic.dense_from_sparse(valid_set_x[index * batch_size: (index + 1) * batch_size, :]).reshape((batch_size, 70, 4))[:,randomized_regions[0][0]:randomized_regions[0][1]],
			},
			on_unused_input='ignore'
		)

		activations = numpy.concatenate([get_layer0_activations(i) for i in xrange(n_batches)], axis=0)

		print('Computed layer activations')

		input_x = input_x[:activations.shape[0],:]
		L_index = L_index[:activations.shape[0]]
		input_x = numpy.asarray(input_x.todense()).reshape((activations.shape[0], self.input_size, self.num_features))[:, 0:self.left_random_size, :]

		y_test_hat = self.get_prediction()
		y_test = test_set_y.eval()[:y_test_hat.shape[0],1]
		logodds_test_hat = safe_log(y_test_hat / (1 - y_test_hat))
		logodds_test = safe_log(y_test / (1 - y_test))

		logodds_test_isinf = numpy.isinf(logodds_test)
		logodds_test_hat = logodds_test_hat[logodds_test_isinf == False]
		logodds_test = logodds_test[logodds_test_isinf == False]
		activations = activations[logodds_test_isinf == False, :, :, :]
		input_x = input_x[logodds_test_isinf == False, :, :]
		L_index = L_index[logodds_test_isinf == False]

		print('Reshaped and filtered activations')

		#logodds_test_hat injection
		logodds_test = logodds_test_hat

		logodds_test_avg = numpy.average(logodds_test)
		logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

		max_activation = numpy.zeros((activations.shape[1], activations.shape[0]))
		max_activation_up = numpy.zeros((activations.shape[1], activations.shape[0]))
		max_activation_pas = numpy.zeros((activations.shape[1], activations.shape[0]))
		max_activation_dn = numpy.zeros((activations.shape[1], activations.shape[0]))
		pos_activation = numpy.zeros((activations.shape[1], activations.shape[0], activations.shape[2]))

		pos_r = numpy.zeros((activations.shape[1], activations.shape[2]))

		filter_width = 8

		valid_activations = numpy.zeros(activations.shape)
		
		#No-padding Library variation strings
		'''libvar_id = numpy.zeros((36, 185 - 7))
		for lib in range(0, 36) :
			lib_index = numpy.nonzero(L_index == lib)[0]
			lib_index_len = len(lib_index)

			
			if lib_index_len > 0 :
				input_x_lib = input_x[lib_index, :, :]
				input_x_lib_sum = numpy.sum(input_x_lib, axis=0)
				constant_base_prev = True
				for j in range(0, input_x_lib_sum.shape[0] - 7) :
					constant_base = False
					for k in range(0, input_x_lib_sum.shape[1]) :
						if input_x_lib_sum[j, k] == lib_index_len :
							constant_base = True
							break
					if constant_base == False :
						libvar_id[lib, j] = 1

					if constant_base == True and constant_base_prev == False and j-7 >= 0 :
						libvar_id[lib, j-7:j] = 0

					constant_base_prev = constant_base

		print('Debug libvar strings:')
		for lib in range(0, 36) :
			lib_index = numpy.nonzero(L_index == lib)[0]
			lib_index_len = len(lib_index)

			if lib_index_len > 0 :
				libvar = ''
				for j in range(0, libvar_id.shape[1]) :
					if libvar_id[lib, j] == 1 :
						libvar = libvar + 'V'
					else :
						libvar = libvar + 'X'
				print(libvar)'''


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

		'''pos_to_libs_lookup = []
		for pos in range(0, libvar_id.shape[1]) :
			valid_libs = []
			valid_libs_str = ''

			for lib in range(0, 36) :
				if libvar_id[lib, pos] == 1 :
					valid_libs.append(lib)
					valid_libs_str += '_' + str(lib)
			pos_to_libs_lookup.append([pos, valid_libs, valid_libs_str])

		for lib in range(0, 36) :
			if numpy.sum(numpy.ravel(libvar_id[lib, :])) > 0 :
				valid_activations[L_index == lib, :, :, :] = numpy.reshape(numpy.tile(libvar_id[lib, :], (len(numpy.nonzero(L_index == lib)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == lib)[0]), activations.shape[1], activations.shape[2], 1))
		'''

		#valid_activations[L_index == 2, :, :, :] = numpy.reshape(numpy.tile(libvar_2_id, (len(numpy.nonzero(L_index == 2)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 2)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 8, :, :, :] = numpy.reshape(numpy.tile(libvar_8_id, (len(numpy.nonzero(L_index == 8)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 8)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 5, :, :, :] = numpy.reshape(numpy.tile(libvar_5_id, (len(numpy.nonzero(L_index == 5)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 5)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 11, :, :, :] = numpy.reshape(numpy.tile(libvar_11_id, (len(numpy.nonzero(L_index == 11)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 11)[0]), activations.shape[1], activations.shape[2], 1))
		valid_activations[L_index == 20, :, :, :] = numpy.reshape(numpy.tile(libvar_20_id, (len(numpy.nonzero(L_index == 20)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 20)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 21, :, :, :] = numpy.reshape(numpy.tile(libvar_21_id, (len(numpy.nonzero(L_index == 21)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 21)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 22, :, :, :] = numpy.reshape(numpy.tile(libvar_22_id, (len(numpy.nonzero(L_index == 22)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 22)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 30, :, :, :] = numpy.reshape(numpy.tile(libvar_30_id, (len(numpy.nonzero(L_index == 30)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 30)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 31, :, :, :] = numpy.reshape(numpy.tile(libvar_31_id, (len(numpy.nonzero(L_index == 31)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 31)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 32, :, :, :] = numpy.reshape(numpy.tile(libvar_32_id, (len(numpy.nonzero(L_index == 32)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 32)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 33, :, :, :] = numpy.reshape(numpy.tile(libvar_33_id, (len(numpy.nonzero(L_index == 33)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 33)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 34, :, :, :] = numpy.reshape(numpy.tile(libvar_34_id, (len(numpy.nonzero(L_index == 34)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 34)[0]), activations.shape[1], activations.shape[2], 1))
		#valid_activations[L_index == 35, :, :, :] = numpy.reshape(numpy.tile(libvar_35_id, (len(numpy.nonzero(L_index == 35)[0]), activations.shape[1], 1)), (len(numpy.nonzero(L_index == 35)[0]), activations.shape[1], activations.shape[2], 1))
		
		
		activations = numpy.multiply(activations, valid_activations)

		k_r = numpy.zeros(activations.shape[1])

		#(num_data_points, num_filters, seq_length, 1)
		for k in range(0, activations.shape[1]) :
			filter_activations = activations[:, k, :, :].reshape((activations.shape[0], activations.shape[2]))
			total_activations = numpy.ravel(numpy.sum(filter_activations, axis=1))

			max_activation[k, :] = numpy.ravel(numpy.max(filter_activations, axis=1))

			#Region-specific max activations
			max_activation_up[k, :] = numpy.ravel(numpy.max(filter_activations[:,:45], axis=1))
			max_activation_pas[k, :] = numpy.ravel(numpy.max(filter_activations[:,45:52], axis=1))
			max_activation_dn[k, :] = numpy.ravel(numpy.max(filter_activations[:,52:97], axis=1))
			max_activation_comp = numpy.ravel(numpy.max(filter_activations[:,97:], axis=1))


			pos_activation[k, :, :] = filter_activations[:, :]

			spike_index = numpy.nonzero(total_activations > 0)[0]

			filter_activations = filter_activations[spike_index, :]

			#print(input_x.shape)
			#print(spike_index.shape)

			filter_inputs = input_x[spike_index, :, :]
			filter_L = L_index[spike_index]

			max_spike = numpy.ravel(numpy.argmax(filter_activations, axis=1))

			max_act = numpy.max(numpy.ravel(numpy.max(filter_activations, axis=1)))

			top_scoring_index = numpy.argsort(numpy.ravel(numpy.max(filter_activations, axis=1)))
			top_scoring_index = top_scoring_index[len(top_scoring_index)-1500:]#5000

			'''PFM = numpy.zeros((filter_width, self.num_features))
			for i in range(0, filter_activations.shape[0]) :
				#if filter_activations[i, max_spike[i]] >= max_act / 2.0 :
				filter_input = filter_inputs[i, max_spike[i]:max_spike[i]+filter_width, :] * filter_activations[i, max_spike[i]]
				PFM = PFM + filter_input'''
			PFM = numpy.zeros((filter_width, self.num_features))
			for ii in range(0, len(top_scoring_index)) :
				i = top_scoring_index[ii]
				#if filter_activations[i, max_spike[i]] >= max_act / 2.0 :
				filter_input = filter_inputs[i, max_spike[i]:max_spike[i]+filter_width, :] #* filter_activations[i, max_spike[i]]
				PFM = PFM + filter_input

			print('Motif ' + str(k))

			#Calculate Pearson r
			logodds_test_curr = logodds_test
			logodds_test_avg_curr = logodds_test_avg
			logodds_test_std_curr = logodds_test_std

			max_activation_regions = [
				[numpy.ravel(max_activation[k, :]), 'All region'],
				[numpy.ravel(max_activation_up[k, :]), 'Upstream'],
				[numpy.ravel(max_activation_pas[k, :]), 'PAS'],
				[numpy.ravel(max_activation_dn[k, :]), 'Downstream'],
				[numpy.ravel(max_activation_comp), 'Distal']
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

				if region[1] == 'All region' :
					k_r[k] = r
				elif region[1] == 'Upstream' :
					r_up_k = r
				elif region[1] == 'PAS' :
					r_pas_k = r
				elif region[1] == 'Downstream' :
					r_dn_k = r

			print('')

			'''max_activation_k = numpy.ravel(max_activation[k, :])

			max_activation_k = max_activation_k[L_index > 5]
			logodds_test_curr = logodds_test[L_index > 5]

			max_activation_k_avg = numpy.average(max_activation_k)
			max_activation_k_std = numpy.sqrt(numpy.dot(max_activation_k - max_activation_k_avg, max_activation_k - max_activation_k_avg))

			logodds_test_avg_curr = numpy.average(logodds_test_curr)
			logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

			cov = numpy.dot(logodds_test_curr - logodds_test_avg_curr, max_activation_k - max_activation_k_avg)
			r = cov / (max_activation_k_std * logodds_test_std_curr)
			print('r = ' + str(round(r, 2)))'''

			prev_selection_libs_str = 'X'
			for pos in range(0, activations.shape[2]) :

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
					
					pos_activation_curr = pos_activation[:, whitelist_index, :]
					logodds_test_curr = logodds_test[whitelist_index]
				logodds_test_avg_curr = numpy.average(logodds_test_curr)
				logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

				if curr_selection_libs_str == '' :
					continue

				pos_activation_k_pos = numpy.ravel(pos_activation_curr[k, :, pos])
				pos_activation_k_pos_avg = numpy.average(pos_activation_k_pos)
				pos_activation_k_pos_std = numpy.sqrt(numpy.dot(pos_activation_k_pos - pos_activation_k_pos_avg, pos_activation_k_pos - pos_activation_k_pos_avg))

				cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k_pos - pos_activation_k_pos_avg)
				r_k_pos = cov_pos / (pos_activation_k_pos_std * logodds_test_std_curr)

				if not (numpy.isinf(r_k_pos) or numpy.isnan(r_k_pos)) :
					pos_r[k, pos] = r_k_pos

				prev_selection_libs_str = curr_selection_libs_str
				pos_activation_prev = pos_activation_curr
				logodds_test_prev = logodds_test_curr

			logo_name = "avg_motif_" + str(k) + ".png"

			logotitle = "Layer 1 Filter " + str(k)
			self.get_logo(k, PFM, 'cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter/' + logo_name, 8, logotitle=logotitle)#r


		'''kk_r = numpy.zeros((activations.shape[1], activations.shape[1]))
		#Estimate combinatorial effects using multiple pearson r
		for k1 in range(0, activations.shape[1]) :
			max_activation_k1 = numpy.ravel(max_activation_up[k1, :])
			#max_activation_k1_avg = numpy.average(max_activation_k1)
			#max_activation_k1_std = numpy.sqrt(numpy.dot(max_activation_k1 - max_activation_k1_avg, max_activation_k1 - max_activation_k1_avg))

			for k2 in range(k1 + 1, activations.shape[1]) :
				#max_activation[k, :] = numpy.ravel(numpy.max(filter_activations, axis=1))

				max_activation_k2 = numpy.ravel(max_activation_dn[k2, :])
				#max_activation_k2_avg = numpy.average(max_activation_k2)
				#max_activation_k2_std = numpy.sqrt(numpy.dot(max_activation_k2 - max_activation_k2_avg, max_activation_k2 - max_activation_k2_avg))

				max_activation_k1_k2 = numpy.multiply(max_activation_k1, max_activation_k2) #+ max_activation_k1 + max_activation_k2

				max_activation_k1_k2_avg = numpy.average(max_activation_k1_k2)
				max_activation_k1_k2_std = numpy.sqrt(numpy.dot(max_activation_k1_k2 - max_activation_k1_k2_avg, max_activation_k1_k2 - max_activation_k1_k2_avg))

				logodds_test_avg = numpy.average(logodds_test)
				logodds_test_std = numpy.sqrt(numpy.dot(logodds_test - logodds_test_avg, logodds_test - logodds_test_avg))

				cov = numpy.dot(logodds_test - logodds_test_avg, max_activation_k1_k2 - max_activation_k1_k2_avg)
				r = cov / (max_activation_k1_k2_std * logodds_test_std)

				kk_r[k1, k2] = r

		k_votes = {}
		for top in range(0, 50) :
			max_k1 = -1
			max_k2 = -1
			max_r = -1
			for k1 in range(0, activations.shape[1]) :
				if k1 not in k_votes :
					k_votes[k1] = 0
				for k2 in range(k1 + 1, activations.shape[1]) :
					if k2 not in k_votes :
						k_votes[k2] = 0
					if kk_r[k1, k2] > max_r and kk_r[k1, k2] > k_r[k1] and kk_r[k1, k2] > k_r[k2] and k_votes[k1] <= 3 and k_votes[k2] <= 3:
						max_r = kk_r[k1, k2]
						max_k1 = k1
						max_k2 = k2

						k_votes[k1] = k_votes[k1] + 1
						k_votes[k2] = k_votes[k2] + 1
			print('Top ' + str(top) + ' combinatorial filters: ' + str(max_k1) + ', ' + str(max_k2) + ', r = ' + str(round(max_r, 2)))



			pos_act_k1 = pos_activation[max_k1, :, :]
			pos_act_k2 = pos_activation[max_k2, :, :]

			max_act_k1 = numpy.ravel(max_activation_up[max_k1, :])
			max_act_k2 = numpy.ravel(max_activation_dn[max_k2, :])

			pos_r_k1_k2 = numpy.zeros((activations.shape[2], activations.shape[2]))

			for pos1 in range(0, activations.shape[2]) :
				pos_activation_k1_pos = numpy.ravel(pos_act_k1[:, pos1])
				for pos2 in range(0, activations.shape[2]) :
					pos_activation_k2_pos = numpy.ravel(pos_act_k2[:, pos2])

					pos_activation_k1_k2_pos = numpy.multiply(pos_activation_k1_pos, pos_activation_k2_pos) #+ pos_activation_k1_pos + pos_activation_k2_pos

					pos_activation_k1_k2_pos = pos_activation_k1_k2_pos#[(max_act_k1 > 0) & (max_act_k2 > 0)]
					logodds_test_curr = logodds_test#[(max_act_k1 > 0) & (max_act_k2 > 0)]
					logodds_test_avg_curr = numpy.average(logodds_test_curr)
					logodds_test_std_curr = numpy.sqrt(numpy.dot(logodds_test_curr - logodds_test_avg_curr, logodds_test_curr - logodds_test_avg_curr))

					pos_activation_k1_k2_pos_avg = numpy.average(pos_activation_k1_k2_pos)
					pos_activation_k1_k2_pos_std = numpy.sqrt(numpy.dot(pos_activation_k1_k2_pos - pos_activation_k1_k2_pos_avg, pos_activation_k1_k2_pos - pos_activation_k1_k2_pos_avg))

					cov_pos = numpy.dot(logodds_test_curr - logodds_test_avg_curr, pos_activation_k1_k2_pos - pos_activation_k1_k2_pos_avg)
					r_k1_k2_pos = cov_pos / (pos_activation_k1_k2_pos_std * logodds_test_std_curr)

					if not (numpy.isinf(r_k1_k2_pos) or numpy.isnan(r_k1_k2_pos)) :
						pos_r_k1_k2[pos1, pos2] = r_k1_k2_pos

			f = plt.figure(figsize=(16, 12))

			plt.pcolor(pos_r_k1_k2,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r_k1_k2).max(), vmax=numpy.abs(pos_r_k1_k2).max())
			plt.colorbar()

			plt.ylabel('Motif ' + str(max_k1) + ' Sequence position')
			plt.ylabel('Motif ' + str(max_k2) + ' Sequence position')
			plt.title('Prox. selection Pearson r for comb. filters')
			#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
			#xticks = mer_sorted
			plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 185], [0 - 49, 25 - 49, 50 - 49, 75 - 49, 100 - 49, 125 - 49, 150 - 49, 175 - 49, 185 - 49])
			plt.yticks([0, 25, 50, 75, 100, 125, 150, 175, 185], [0 - 49, 25 - 49, 50 - 49, 75 - 49, 100 - 49, 125 - 49, 150 - 49, 175 - 49, 185 - 49])

			plt.axis([0, pos_r_k1_k2.shape[1], 0, pos_r_k1_k2.shape[0]])

			#plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter/' + "r_pos_apa_fr.png")
			plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter/' + "r_pos_" + str(max_k1) + "_" + str(max_k2) + ".png")
			plt.close()

			kk_r[max_k1, max_k2] = -1'''


		#All-filter positional Pearson r
		f = plt.figure(figsize=(32, 16))

		plt.pcolor(pos_r,cmap=cm.RdBu_r,vmin=-numpy.abs(pos_r).max(), vmax=numpy.abs(pos_r).max())
		plt.colorbar()

		plt.xlabel('Sequence position')
		plt.title('Prox. selection Pearson r for all filters')
		#plt.axis([0, 4095, np.min(w_sorted) - 0.1, np.max(w_sorted) + 0.1])
		#xticks = mer_sorted
		plt.xticks([0, 25, 50, 75, 100, 125, 150, 175, 185], [0 - 49, 25 - 49, 50 - 49, 75 - 49, 100 - 49, 125 - 49, 150 - 49, 175 - 49, 185 - 49])
		plt.yticks(numpy.arange(pos_r.shape[0]) + 0.5, numpy.arange(pos_r.shape[0]))#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

		plt.axis([0, pos_r.shape[1], 0, pos_r.shape[0]])

		#plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter/' + "r_pos_apa_fr.png")
		plt.savefig('cnn_motif_analysis/fullseq_global_onesided/deconv/avg_filter/' + "r_pos.png")
		plt.close()

	def get_logo(self, k, PFM, file_path='cnn_motif_analysis/fullseq_global_onesided/', seq_length=6, normalize=False, logotitle="") :

		if normalize == True :
			for i in range(0, PFM.shape[0]) :
				if numpy.sum(PFM[i, :]) > 0 :
					PFM[i, :] = PFM[i, :] / numpy.sum(PFM[i, :])
				#PFM[i, :] *= 10000.0
			#print(PFM)

		#Create weblogo from API
		logo_output_format = "png"
		#Load data from an occurence matrix
		data = weblogolib.LogoData.from_counts('ACGT', PFM)

		#Generate color scheme
		colors = weblogolib.ColorScheme([
		        weblogolib.ColorGroup("A", "yellow","CFI Binder" ),
		        weblogolib.ColorGroup("C", "green","CFI Binder" ),
		        weblogolib.ColorGroup("G", "red","CFI Binder" ),
		        weblogolib.ColorGroup("T", "blue","CFI Binder" )] )

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
			poolsize=(2, 1),
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
			n_in=nkerns[1] * (84) * 1 + 1,
			n_out=400,
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

		# classify the values of the fully-connected sigmoidal layer
		layer4 = LogisticRegression(input=layer4_input, L_input=L_input, n_in=400 + 36, n_out=self.input_size + 1, load_model = load_model,
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
			patience = 120000#140000  # look as this many examples regardless
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
		# end-snippet-1

def letterAt(letter, x, y, yscale=1, ax=None, color=None):

	#fp = FontProperties(family="Arial", weight="bold")
	fp = FontProperties(family="Ubuntu", weight="bold")
	globscale = 1.35
	LETTERS = {	"T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
				"G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
				"A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
				"C" : TextPath((-0.366, 0), "C", size=1, prop=fp),
				"UP" : TextPath((-0.488, 0), '$\\Uparrow$', size=1, prop=fp),
				"DN" : TextPath((-0.488, 0), '$\\Downarrow$', size=1, prop=fp) }
	COLOR_SCHEME = {'G': 'orange', 
					'A': 'red', 
					'C': 'blue', 
					'T': 'darkgreen',
					'UP': 'green', 
					'DN': 'red'}


	text = LETTERS[letter]

	chosen_color = COLOR_SCHEME[letter]
	if color is not None :
		chosen_color = color

	t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
		mpl.transforms.Affine2D().translate(x,y) + ax.transData
	p = PathPatch(text, lw=0, fc=chosen_color,  transform=t)
	if ax != None:
		ax.add_artist(p)
	return p

def cut_map(cnn, test_set_x, test_set_y, test_set_L, test_set_d, test_set_c) :

	positions = numpy.arange(0, 186).tolist()

	library = 'All_scatter'#'Simple_scatter'#'WHAMMP2_scatter'#'HSPE1_scatter'#'SNHG6_scatter'#'SOX13_scatter'#'ATR_scatter'#'TOMM5_scatter'#'WHAMMP2'#'Simple'#'SNHG6'

	X = numpy.array(test_set_x.eval().todense())
	print(X.shape)

	X = X.reshape((X.shape[0], X.shape[1] / 4, 4))

	y_cuts = numpy.array(test_set_y.eval().todense())
	y_hat_cuts = numpy.array(cnn.get_prediction_distrib(positions))
	print(y_cuts.shape)
	print(y_hat_cuts.shape)

	c = numpy.ravel(test_set_c.eval())


	y_hat_avgcut = numpy.zeros(X.shape[0])
	y_hat_stdcut = numpy.zeros(X.shape[0])
	y_avgcut = numpy.zeros(X.shape[0])
	y_stdcut = numpy.zeros(X.shape[0])

	start_cut_pos = 60#57 + 7#+ 5
	end_cut_pos = 100#57 + 43#+ 43

	valid_index = []

	for i in range(0, X.shape[0]) :

		seq = translate_matrix_to_seq(X[i, :, :])

		y_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_c[:] = numpy.ravel(y_cuts[i, :])[start_cut_pos:end_cut_pos+1]

		if numpy.sum(y_c) > 0.25 :
			y_c = y_c / numpy.sum(y_c)
			valid_index.append(1)
		else :
			valid_index.append(0)
		
		y_hat_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_hat_c[:] = numpy.ravel(y_hat_cuts[i, :])[start_cut_pos:end_cut_pos+1]

		y_hat_c = y_hat_c / numpy.sum(y_hat_c)

		avg_c = numpy.dot(y_c, numpy.arange(len(y_c)))
		std_c = numpy.sqrt(numpy.dot(y_c, numpy.power(numpy.arange(len(y_c)) - avg_c, 2)))

		avg_hat_c = numpy.dot(y_hat_c, numpy.arange(len(y_hat_c)))
		std_hat_c = numpy.sqrt(numpy.dot(y_hat_c, numpy.power(numpy.arange(len(y_hat_c)) - avg_hat_c, 2)))

		y_avgcut[i] = avg_c
		y_stdcut[i] = std_c
		y_hat_avgcut[i] = avg_hat_c
		y_hat_stdcut[i] = std_hat_c

	valid_index = numpy.nonzero(numpy.ravel(numpy.array(valid_index)))[0]
	y_avgcut = y_avgcut[valid_index]
	y_stdcut = y_stdcut[valid_index]
	y_hat_avgcut = y_hat_avgcut[valid_index]
	y_hat_stdcut = y_hat_stdcut[valid_index]
	X = X[valid_index, :, :]
	c = c[valid_index]



	y_hat_avgcut_avg = numpy.average(y_hat_avgcut)
	y_hat_avgcut_std = numpy.sqrt(numpy.dot(y_hat_avgcut - y_hat_avgcut_avg, y_hat_avgcut - y_hat_avgcut_avg))

	y_avgcut_avg = numpy.average(y_avgcut)
	y_avgcut_std = numpy.sqrt(numpy.dot(y_avgcut - y_avgcut_avg, y_avgcut - y_avgcut_avg))

	cov = numpy.dot(y_hat_avgcut - y_hat_avgcut_avg, y_avgcut - y_avgcut_avg)
	test_r = cov / (y_hat_avgcut_std * y_avgcut_std)

	rsquare = test_r * test_r

	print('Avg Cut R^2 = ' + str(rsquare))

	sort_index = numpy.argsort(y_avgcut)



	fig = plt.figure(figsize=(8, 6))

	c_filter = 20#200


	y_hat_avgcut_avg = numpy.average(y_hat_avgcut[sort_index][c[sort_index] > c_filter])
	y_hat_avgcut_std = numpy.sqrt(numpy.dot(y_hat_avgcut[sort_index][c[sort_index] > c_filter] - y_hat_avgcut_avg, y_hat_avgcut[sort_index][c[sort_index] > c_filter] - y_hat_avgcut_avg))

	y_avgcut_avg = numpy.average(y_avgcut[sort_index][c[sort_index] > c_filter])
	y_avgcut_std = numpy.sqrt(numpy.dot(y_avgcut[sort_index][c[sort_index] > c_filter] - y_avgcut_avg, y_avgcut[sort_index][c[sort_index] > c_filter] - y_avgcut_avg))

	cov = numpy.dot(y_hat_avgcut[sort_index][c[sort_index] > c_filter] - y_hat_avgcut_avg, y_avgcut[sort_index][c[sort_index] > c_filter] - y_avgcut_avg)
	test_r = cov / (y_hat_avgcut_std * y_avgcut_std)

	rsquare_filtered = test_r * test_r

	print('Filtered Avg Cut R^2 = ' + str(rsquare_filtered))



	used_size = len(numpy.nonzero(c[sort_index] > c_filter)[0])

	plt.scatter(y_hat_avgcut[sort_index][c[sort_index] > c_filter], numpy.arange(len(sort_index[c[sort_index] > c_filter])), s=numpy.pi * (2 * numpy.ones(1))**2, color='darkblue', alpha=0.25)
	#plt.plot(y_hat_avgcut[sort_index][c[sort_index] > c_filter], numpy.arange(len(sort_index[c[sort_index] > c_filter])), color='darkblue', linestyle='-', alpha=0.5, linewidth=2)
	plt.plot(y_avgcut[sort_index][c[sort_index] > c_filter], numpy.arange(len(sort_index[c[sort_index] > c_filter])), color='red', linestyle='--', alpha=0.8, linewidth=4)

	plt.axis([0, max(numpy.max(y_avgcut[sort_index][c[sort_index] > c_filter]), numpy.max(y_hat_avgcut[sort_index][c[sort_index] > c_filter])), 0, len(sort_index[c[sort_index] > c_filter])])

	plt.savefig('cnn_cuts_predicted_vs_target_' + library + '_r2_' + str(round(rsquare, 2)) + '_size_' + str(used_size) + '.png')
	plt.savefig('cnn_cuts_predicted_vs_target_' + library + '_r2_' + str(round(rsquare, 2)) + '_size_' + str(used_size) + '_vector.svg')
	plt.savefig('cnn_cuts_predicted_vs_target_' + library + '_r2_' + str(round(rsquare, 2)) + '_size_' + str(used_size) + '_vector.eps')
	plt.show()
	plt.close()


	print('' + 1)

	#Extract interesting avg cut outliers


	start_cut_pos = 39#57 + 7#10#+ 5
	end_cut_pos = 100#57 + 43#33#+ 43

	#Extract interesting std cut outliers

	#sort_index_stdcut = numpy.argsort(y_stdcut)[::-1]
	sort_index_stdcut = numpy.argsort(numpy.ravel(numpy.sum(numpy.power(y_cuts[:, start_cut_pos:end_cut_pos+1] - y_hat_cuts[:, start_cut_pos:end_cut_pos+1], 2), axis=1)))

	print('Stdcut.')

	min_spike = 0.2#0.05#0.15#0.2
	min_spike_distance = 7#2#7#7
	min_spike_num = 2#4#2#2

	ii = 0
	ii_used = 0
	while ii_used < 20 :

		i = sort_index_stdcut[ii]

		seq = translate_matrix_to_seq(X[i, :, :])[start_cut_pos:end_cut_pos+1]

		y_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_c[:] = numpy.ravel(y_cuts[i, :])[start_cut_pos:end_cut_pos+1]
		y_hat_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_hat_c[:] = numpy.ravel(y_hat_cuts[i, :])[start_cut_pos:end_cut_pos+1]

		spike_index = numpy.nonzero(y_hat_c > min_spike)[0]
		red_spike_index = []
		for j in range(0, len(spike_index)) :
			if len(red_spike_index) == 0 :
				red_spike_index.append(spike_index[j])
			elif spike_index[j] - red_spike_index[len(red_spike_index) - 1] > min_spike_distance :
				red_spike_index.append(spike_index[j])
			else :
				red_spike_index[len(red_spike_index) - 1] = spike_index[j]

		if numpy.sum(y_c) < 0.2 or c[i] < 200 or len(red_spike_index) < min_spike_num :
			ii += 1
			continue

		full_seq = translate_matrix_to_seq(X[i, :, :])
		if 'AATAAA' in full_seq[:49] or 'ATTAAA' in full_seq[:49] :
			ii += 1
			continue
		if 'AATAAA' in full_seq[55:95] or 'ATTAAA' in full_seq[55:95] :
			ii += 1
			continue

		fig = plt.figure(figsize=(14, 6)) 
		gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax = [ax0, ax1]

		for j in range(0, len(seq)) :
			letterAt(seq[j], j + 0.5, 0, 1.0, ax[1])

		plt.sca(ax[1])
		plt.xlim((0, end_cut_pos+1-start_cut_pos))
		plt.ylim((0, 1))

		ax[0].plot(numpy.arange(len(seq)), y_hat_c, color='darkblue', linestyle='-', linewidth=3, alpha=0.7)
		ax[0].plot(numpy.arange(len(seq)), y_c, color='darkred', linestyle='--', linewidth=3, alpha=0.7)

		ax[0].plot([10, 10], [0, numpy.max(y_hat_c)], color='green', linestyle='--', linewidth=3)
		ax[0].plot([10+6, 10+6], [0, numpy.max(y_hat_c)], color='green', linestyle='--', linewidth=3)

		plt.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')



		plt.sca(ax[0])
		plt.xticks([0, 10, 20, 30, 40, 50, 60], [-10, 0, 10, 20, 30, 40, 50])
		#plt.xticks(numpy.arange(len(seq)), numpy.arange(len(seq)) + 39 - 49)
		plt.xlim((0, end_cut_pos+1-start_cut_pos))
		plt.ylim((0, numpy.max(y_hat_c)))

		plt.tight_layout()

		plt.savefig('cnn_cuts_stdcut_' + library + '_example_' + str(int(sort_index_lookup[i])) + '.png', bbox_inches='tight')
		plt.savefig('cnn_cuts_stdcut_' + library + '_example_' + str(int(sort_index_lookup[i])) + '_vector.svg', bbox_inches='tight')
		plt.show()
		plt.close()

		ii += 1
		ii_used += 1
	


	print('Avgcut.')

	'''sort_index_stdcut = numpy.argsort(numpy.ravel(numpy.sum(numpy.power(y_cuts[:, start_cut_pos:end_cut_pos+1] - y_hat_cuts[:, start_cut_pos:end_cut_pos+1], 2), axis=1)))

	min_spike = 0.1
	min_spike_distance = 7#7#7
	max_spike_num = 1#2#2

	max_distance = 40

	ii = 0
	ii_used = 0
	while ii_used < 20 :

		i = sort_index_stdcut[ii]

		seq = translate_matrix_to_seq(X[i, :, :])[start_cut_pos:end_cut_pos+1]

		y_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_c[:] = numpy.ravel(y_cuts[i, :])[start_cut_pos:end_cut_pos+1]
		y_hat_c = numpy.zeros(end_cut_pos+1-start_cut_pos)
		y_hat_c[:] = numpy.ravel(y_hat_cuts[i, :])[start_cut_pos:end_cut_pos+1]

		spike_index = numpy.nonzero(y_hat_c > min_spike)[0]
		red_spike_index = []
		for j in range(0, len(spike_index)) :
			if len(red_spike_index) == 0 :
				red_spike_index.append(spike_index[j])
			elif spike_index[j] - red_spike_index[len(red_spike_index) - 1] > min_spike_distance :
				red_spike_index.append(spike_index[j])
			else :
				red_spike_index[len(red_spike_index) - 1] = spike_index[j]

		if numpy.sum(y_c) < 0.2 or c[i] < 200 or len(red_spike_index) != 1 :
			ii += 1
			continue

		if red_spike_index[0] < max_distance :
			ii += 1
			continue

		full_seq = translate_matrix_to_seq(X[i, :, :])
		if 'AATAAA' in full_seq[:49] or 'ATTAAA' in full_seq[:49] :
			ii += 1
			continue
		if 'AATAAA' in full_seq[55:95] or 'ATTAAA' in full_seq[55:95] :
			ii += 1
			continue

		fig = plt.figure(figsize=(14, 6)) 
		gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

		ax0 = plt.subplot(gs[0])
		ax1 = plt.subplot(gs[1])
		ax = [ax0, ax1]

		for j in range(0, len(seq)) :
			letterAt(seq[j], j + 0.5, 0, 1.0, ax[1])

		plt.sca(ax[1])
		plt.xlim((0, end_cut_pos+1-start_cut_pos))
		plt.ylim((0, 1))

		ax[0].plot(numpy.arange(len(seq)), y_hat_c, color='darkblue', linestyle='-', linewidth=3, alpha=0.7)
		ax[0].plot(numpy.arange(len(seq)), y_c, color='darkred', linestyle='--', linewidth=3, alpha=0.7)

		ax[0].plot([10, 10], [0, numpy.max(y_hat_c)], color='green', linestyle='--', linewidth=3)
		ax[0].plot([10+6, 10+6], [0, numpy.max(y_hat_c)], color='green', linestyle='--', linewidth=3)

		plt.tick_params(
			axis='x',          # changes apply to the x-axis
			which='both',      # both major and minor ticks are affected
			bottom='off',      # ticks along the bottom edge are off
			top='off',         # ticks along the top edge are off
			labelbottom='off')



		plt.sca(ax[0])
		plt.xticks([0, 10, 20, 30, 40, 50, 60], [-10, 0, 10, 20, 30, 40, 50])
		#plt.xticks(numpy.arange(len(seq)), numpy.arange(len(seq)) + 39 - 49)
		plt.xlim((0, end_cut_pos+1-start_cut_pos))
		plt.ylim((0, numpy.max(y_hat_c)))

		plt.tight_layout()

		plt.savefig('cnn_cuts_maxcut_' + library + '_example_' + str(int(sort_index_lookup[i])) + '.png', bbox_inches='tight')
		plt.savefig('cnn_cuts_maxcut_' + library + '_example_' + str(int(sort_index_lookup[i])) + '_vector.svg', bbox_inches='tight')
		plt.show()
		plt.close()

		ii += 1
		ii_used += 1'''


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))


def evaluate_cnn(dataset='cenpb1varcuts2'):

	dataset = 'general_cuts_antimisprime_orig'#'cenpb1varcuts'#'general_cuts_antimisprime_orig'#'cenpb1varcuts2'#_ALIGNED

	count_filter = 0

	input_datasets = load_input_data(dataset, shuffle=True, count_filter=count_filter, balance_all_libraries=True)
	
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

	train_set_c = input_datasets[11]
	valid_set_c = input_datasets[12]
	test_set_c = input_datasets[13]

	output_datasets = load_output_data(dataset, shuffle_index, count_filter=count_filter, balance_all_libraries=True)
	
	train_set_y = output_datasets[0]
	valid_set_y = output_datasets[1]
	test_set_y = output_datasets[2]

	batch_size = 1

	
	cnn = CutCNN(
		(train_set_x, train_set_y, train_set_L, train_set_d),
		(valid_set_x, valid_set_y, valid_set_L, valid_set_d),
		learning_rate=0.1,
		drop=0.2,
		n_epochs=10,
		nkerns=[70, 110, 70],
		batch_size=batch_size,
		num_features=4,
		#randomized_regions=[(75 + 4, 260 + 4), (260 + 7, 260 + 7)],#185# + 2, + 3, + 9, + 10,,,,,, + 2 / + 4
		#randomized_regions=[(75 + 1, 260 + 1), (260 + 7, 260 + 7)],#+7
		randomized_regions=[(0, 185), (185, 185)],
		load_model=True,
		train_model_flag=False,
		store_model=False,
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorig_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorig_finetuned_TOMM5_APA_Six_30_31_34'
		dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
		store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34'
	)
	
	print('Trained sublib bias terms:')
	lrW = cnn.output_layer.W.eval()
	lrW = numpy.ravel(lrW[lrW.shape[0] - 36:, 1])
	for i in range(0, len(lrW)) :
		if lrW[i] != 0 :
			print(str(i) + ": " + str(lrW[i]))


	cnn.set_data(test_set_x, test_set_y, test_set_L, test_set_d)
	
	cut_map(cnn, test_set_x, test_set_y, test_set_L, test_set_d, test_set_c)



def safe_log(x, minval=0.02):
    return numpy.log(x.clip(min=minval))

def logit(x) :
	return numpy.log(x / (1.0 - x))

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
	evaluate_cnn('general')
