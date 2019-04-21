
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

from logistic_sgd_global_onesided2_cuts_snp import LogisticRegression #, load_input_data, load_output_data
from logistic_sgd_global_onesided2_geuv import load_input_data, load_output_data #, LogisticRegression

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

def align_predict_snps(cnn, ref_x, ref_y, ref_d, var_x, var_y, L_zero, var_d, apadist, snptype, run_name) :

	n = ref_y.eval().shape[0]

	ref_x = numpy.array(ref_x.eval().todense())
	ref_x = ref_x.reshape((ref_x.shape[0], ref_x.shape[1] / 4, 4))

	var_x = numpy.array(var_x.eval().todense())
	var_x = var_x.reshape((var_x.shape[0], var_x.shape[1] / 4, 4))

	L_zero = L_zero.eval()

	ref_y = ref_y.eval()
	var_y = var_y.eval()

	ref_d = ref_d.eval()
	var_d = var_d.eval()


	y_ref = logit(ref_y[:,1])
	y_var = logit(var_y[:,1])
	diff = numpy.ravel(y_var) - numpy.ravel(y_ref)


	align_min = -25#-10#-25
	align_max = 5#3#5

	diff_hat = []

	ref_snp_strs = []
	var_snp_strs = []

	aligned_ref_x = numpy.zeros(ref_x.shape)
	aligned_var_x = numpy.zeros(var_x.shape)

	aligned_ref_seqs = []
	aligned_var_seqs = []


	valid_pas = [
		'AATAAA',
		'ATTAAA',
		'AGTAAA',
		'TATAAA',
		'GATAAA',
		'CATAAA',
		'ATATAA',
		'AAGAAA',
		'AACAAA',
		'AAAAAA',
		'CAAAAA',
		'TAAAAA',
		'GAAAAA',
		'AAATAA',
		'AAGTAA',
		'AATAAT',
		'TAATAA',
		'TAAGAA',
		'GAATAA',
		'CAATAA',
		'AATAAG',
		'AATAAC',
		'AATGAA',
		'ATGAAT',
		#'AATGAG'#
	]

	cano_pas1 = 'AATAAA'
	cano_pas2 = 'ATTAAA'

	for pos in range(0, 6) :
		for base in ['A', 'C', 'G', 'T'] :
			if cano_pas1[:pos] + base + cano_pas1[pos+1:] not in valid_pas :
				valid_pas.append(cano_pas1[:pos] + base + cano_pas1[pos+1:])
			if cano_pas2[:pos] + base + cano_pas2[pos+1:] not in valid_pas :
				valid_pas.append(cano_pas2[:pos] + base + cano_pas2[pos+1:])

	for pos1 in range(0, 6) :
		for pos2 in range(pos1 + 1, 6) :
			for base1 in ['A', 'C', 'G', 'T'] :
				for base2 in ['A', 'C', 'G', 'T'] :
					if cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:] not in valid_pas :
						valid_pas.append(cano_pas1[:pos1] + base1 + cano_pas1[pos1+1:pos2] + base2 + cano_pas1[pos2+1:])
					#if cano_pas2[:pos1] + base1 + cano_pas2[pos1+1:pos2] + base2 + cano_pas2[pos2+1:] not in valid_pas :
					#	valid_pas.append(cano_pas2[:pos1] + base1 + cano_pas2[pos1+1:pos2] + base2 + cano_pas2[pos2+1:])'''

	aligned_ref_x = numpy.zeros(ref_x.shape)
	aligned_var_x = numpy.zeros(var_x.shape)


	for i in range(0, n) :

		best_align_hat = -4
		best_align_j = 0
		best_align_diff_hat = 0
		for j in range(align_min, align_max) :

			ref_x_curr = numpy.zeros((1, ref_x.shape[1], 4))
			ref_x_curr[:, :, :] = ref_x[i, :, :]

			if j < 0 :
				align = int(numpy.abs(j))
				ref_x_curr = numpy.concatenate([ref_x_curr[:, align:, :], numpy.zeros((1, align, 4))], axis=1)

			if j > 0 :
				align = int(numpy.abs(j))
				ref_x_curr = numpy.concatenate([numpy.zeros((1, align, 4)), ref_x_curr[:, :ref_x.shape[1]-align, :]], axis=1)

			#ref_x_curr = theano.shared(sp.csr_matrix(ref_x_curr.reshape((1, ref_x.shape[1] * 4))), borrow=True)
			#ref_x_curr = sp.csr_matrix(ref_x_curr.reshape((1, ref_x.shape[1] * 4)))

			L_zero_curr = numpy.zeros((1, 36))
			L_zero_curr[:, :] = L_zero[i, :]
			#L_zero_curr = theano.shared(L_zero_curr, borrow=True)

			ref_y_curr = numpy.zeros((1, 2))
			ref_y_curr[:, :] = ref_y[i, :]
			#ref_y_curr = theano.shared(ref_y_curr, borrow=True)

			ref_d_curr = numpy.zeros((1, 1))
			ref_d_curr[:, :] = ref_d[i, :]
			#ref_d_curr = theano.shared(ref_d_curr, borrow=True)


			#cnn.set_data(ref_x_curr, ref_y_curr, L_zero_curr, ref_d_curr)
			#y_ref_hat = logit(cnn.get_prediction())
			y_ref_hat = logit(cnn.get_online_prediction(ref_x_curr, L_zero_curr, ref_d_curr))[0]


			var_x_curr = numpy.zeros((1, var_x.shape[1], 4))
			var_x_curr[:, :, :] = var_x[i, :, :]

			if j < 0 :
				align = int(numpy.abs(j))
				var_x_curr = numpy.concatenate([var_x_curr[:, align:, :], numpy.zeros((1, align, 4))], axis=1)

			if j > 0 :
				align = int(numpy.abs(j))
				var_x_curr = numpy.concatenate([numpy.zeros((1, align, 4)), var_x_curr[:, :var_x.shape[1]-align, :]], axis=1)

			#var_x_curr = theano.shared(sp.csr_matrix(var_x_curr.reshape((1, var_x.shape[1] * 4))), borrow=True)

			var_y_curr = numpy.zeros((1, 2))
			var_y_curr[:, :] = var_y[i, :]
			#var_y_curr = theano.shared(var_y_curr, borrow=True)

			var_d_curr = numpy.zeros((1, 1))
			var_d_curr[:, :] = var_d[i, :]
			#var_d_curr = theano.shared(var_d_curr, borrow=True)


			#cnn.set_data(var_x_curr, var_y_curr, L_zero_curr, var_d_curr)
			#y_var_hat = logit(cnn.get_prediction())

			y_var_hat = logit(cnn.get_online_prediction(var_x_curr, L_zero_curr, var_d_curr))[0]


			ref_x_seq = translate_matrix_to_seq(ref_x_curr[0, :, :])#[75+1:]
			var_x_seq = translate_matrix_to_seq(var_x_curr[0, :, :])#[75+1:]

			if y_ref_hat > best_align_hat or y_var_hat > best_align_hat :

				if ref_x_seq[49:49+6] in valid_pas or var_x_seq[49:49+6] in valid_pas :
					#best_align_hat = y_ref_hat
					best_align_hat = max(y_ref_hat, y_var_hat)

					best_align_j = j
					best_align_diff_hat = y_var_hat - y_ref_hat
		diff_hat.append(best_align_diff_hat)


		print('Aligned member ' + str(i) + '(d = ' + str(apadist[i]) + '), align = ' + str(best_align_j))

		print('pas: ' + '                                                 ' + '|')

		ref_x_curr = numpy.zeros((ref_x.shape[1], 4))
		ref_x_curr[:, :] = ref_x[i, :, :]

		if best_align_j < 0 :
			align = int(numpy.abs(best_align_j))
			ref_x_curr = numpy.concatenate([ref_x_curr[align:, :], numpy.zeros((align, 4))], axis=0)
		if best_align_j > 0 :
			align = int(numpy.abs(best_align_j))
			ref_x_curr = numpy.concatenate([numpy.zeros((align, 4)), ref_x_curr[:ref_x.shape[1]-align, :]], axis=0)
		#print('ref: ' + translate_matrix_to_seq(ref_x_curr)[75+1:])#75+1+90])
		
		ref_seq = translate_matrix_to_seq(ref_x_curr)#[75+1:]

		var_x_curr = numpy.zeros((var_x.shape[1], 4))
		var_x_curr[:, :] = var_x[i, :, :]

		if best_align_j < 0 :
			align = int(numpy.abs(best_align_j))
			var_x_curr = numpy.concatenate([var_x_curr[align:, :], numpy.zeros((align, 4))], axis=0)
		if best_align_j > 0 :
			align = int(numpy.abs(best_align_j))
			var_x_curr = numpy.concatenate([numpy.zeros((align, 4)), var_x_curr[:var_x.shape[1]-align, :]], axis=0)
		#print('var: ' + translate_matrix_to_seq(var_x_curr)[75+1:])#75+1+90])

		var_seq = translate_matrix_to_seq(var_x_curr)#[75+1:]

		aligned_ref_x[i, :, :] = ref_x_curr[:, :]
		aligned_ref_seqs.append(translate_matrix_to_seq(ref_x_curr))
		aligned_var_x[i, :, :] = var_x_curr[:, :]
		aligned_var_seqs.append(translate_matrix_to_seq(var_x_curr))

		ref_seq_snp = ''
		var_seq_snp = ''
		for k in range(0, len(ref_seq)) :
			if ref_seq[k] != var_seq[k] :
				ref_seq_snp = ref_seq[k-15:k+16]
				ref_seq_snp += ' (PAS '
				if k >= 49 :
					ref_seq_snp += '+ ' + str(int(numpy.abs(k - 49))) + ')'
				else :
					ref_seq_snp += '- ' + str(int(numpy.abs(k - 49))) + ')'

				var_seq_snp = (' ' * 15) + var_seq[k]

				break
		ref_snp_strs.append(ref_seq_snp)
		var_snp_strs.append(var_seq_snp)



	for i in range(0, n) :
		print(i)
		print(ref_snp_strs[i])
		print((' ' * 15) + '^')
		print(var_snp_strs[i])


	diff_hat = numpy.ravel(numpy.array(diff_hat))

	snp_index = numpy.arange(len(diff_hat))

	'''over_thres_index = numpy.abs(diff_hat) > 0.05
	diff = diff[over_thres_index]
	diff_hat = diff_hat[over_thres_index]
	apadist = apadist[over_thres_index]
	snp_index = snp_index[over_thres_index]'''


	SSE_diff = (diff - diff_hat).T.dot(diff - diff_hat)

	y_diff_average = numpy.average(diff, axis=0)

	SStot_diff = (diff - y_diff_average).T.dot(diff - y_diff_average)

	RMSE_diff = numpy.sqrt(SSE_diff / float(len(y_ref)))

	MAE_diff = numpy.mean(numpy.abs(diff_hat - diff))

	diff_set_dir_accuracy = numpy.count_nonzero(numpy.sign(diff) == numpy.sign(diff_hat))


	print("")
	print("Logodds diff R^2:")
	print(1.0 - (SSE_diff / SStot_diff))
	print("Logodds diff mean abs error:")
	print(MAE_diff)

	print("Logodds diff Classification accuracy: " + str(diff_set_dir_accuracy) + "/" + str(len(diff)) + " = " + str(float(diff_set_dir_accuracy) / float(len(diff))))

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	if len(snptype[snptype == 1]) > 0 :
		ax1.scatter(diff_hat[snptype == 1], diff[snptype == 1], label='Heterozygous', marker='o', c='orange', s=90, alpha=0.5)
	if len(snptype[snptype == 2]) > 0 :
		ax1.scatter(diff_hat[snptype == 2], diff[snptype == 2], label='Homozygous', marker='o', c='red', s=90, alpha=0.5)

	#col = ax1.scatter(diff_hat, diff, c = 'red', s = 4 * numpy.pi * (2 * numpy.ones(1))**2, alpha=0.4)
	ax1.plot([-5, 5], [-5, 5], c='yellow')
	ax1.plot([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1], [0, 0], c='green')
	ax1.plot([0, 0], [numpy.min(diff) * 1.1, numpy.max(diff) * 1.1], c='green')
	ax1.set_xlim([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1])
	ax1.set_ylim([numpy.min(diff) * 1.1, numpy.max(diff) * 1.1])
	ax1.set_xlabel('Predicted Proximal $\Delta$Logodds', fontsize=18)
	ax1.set_ylabel('Observed Proximal $\Delta$Logodds', fontsize=18)
	#ax1.set_title('GEUV APA SNP Log Diff (R^2 = ' + str(round(1.0 - (SSE_diff / SStot_diff), 2)) + ', Acc = ' + str(diff_set_dir_accuracy) + "/" + str(len(diff)) + ')', fontsize=18)
	ax1.set_title('R^2 = ' + str(round(1.0 - (SSE_diff / SStot_diff), 2)) + ', Acc = ' + str(diff_set_dir_accuracy) + "/" + str(len(diff)), fontsize=28)
	
	ax1.legend()

	fig.suptitle('GEUVADIS APA SNPs', fontsize=24)

	for i in range(0, len(diff)):
		#ax1.annotate(snp_index[i] + 2, (diff_hat[i], diff[i]))

		if numpy.abs(diff_hat[i]) > 1.5 or numpy.abs(diff[i]) > 1.5 :

			annotation = '(' + str(int(i)) + ')'
			ax1.annotate(annotation, (diff_hat[i], diff[i]), size=10)

	fig.tight_layout()
		
	plt.subplots_adjust(top=0.83, wspace = 0.6)

	plt.savefig("cnn_cuts_snp_logodds_diff_global" + run_name + "_annotated.svg")
	plt.savefig("cnn_cuts_snp_logodds_diff_global" + run_name + "_annotated.png")
	plt.show()
	plt.close()



	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	if len(snptype[snptype == 1]) > 0 :
		ax1.scatter(diff_hat[snptype == 1], diff[snptype == 1], label='Heterozygous', marker='o', c='orange', s=90, alpha=0.5)
	if len(snptype[snptype == 2]) > 0 :
		ax1.scatter(diff_hat[snptype == 2], diff[snptype == 2], label='Homozygous', marker='o', c='red', s=90, alpha=0.5)

	#col = ax1.scatter(diff_hat, diff, c = 'red', s = 4 * numpy.pi * (2 * numpy.ones(1))**2, alpha=0.4)
	ax1.plot([-5, 5], [-5, 5], c='yellow')
	ax1.plot([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1], [0, 0], c='green')
	ax1.plot([0, 0], [numpy.min(diff) * 1.1, numpy.max(diff) * 1.1], c='green')
	ax1.set_xlim([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1])
	ax1.set_ylim([numpy.min(diff) * 1.1, numpy.max(diff) * 1.1])
	ax1.set_xlabel('Predicted Proximal $\Delta$Logodds', fontsize=18)
	ax1.set_ylabel('Observed Proximal $\Delta$Logodds', fontsize=18)
	#ax1.set_title('GEUV APA SNP Log Diff (R^2 = ' + str(round(1.0 - (SSE_diff / SStot_diff), 2)) + ', Acc = ' + str(diff_set_dir_accuracy) + "/" + str(len(diff)) + ')', fontsize=18)
	ax1.set_title('R^2 = ' + str(round(1.0 - (SSE_diff / SStot_diff), 2)) + ', Acc = ' + str(diff_set_dir_accuracy) + "/" + str(len(diff)), fontsize=28)
	
	ax1.legend()

	fig.suptitle('GEUVADIS APA SNPs', fontsize=24)

	fig.tight_layout()
		
	plt.subplots_adjust(top=0.83, wspace = 0.6)

	plt.savefig("cnn_cuts_snp_logodds_diff_global" + run_name + ".svg")
	plt.savefig("cnn_cuts_snp_logodds_diff_global" + run_name + ".png")
	plt.show()
	plt.close()

	return aligned_ref_seqs, aligned_var_seqs, diff_hat, diff

def mut_map(cnn, ref_seq, name, plot_start=0, plot_end=185, downscale_factor=1.0) :

	ref_x = numpy.zeros((1, len(ref_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x[0, j, 3] = 1
		else :
			ref_x[0, j, :] = 0.25

	#L_zero = numpy.zeros((1, 36))
	#ref_d = numpy.ones((1, 1))
	L_zero = numpy.zeros((1, 36))
	#L_zero[0, 8] = 1
	#L_zero[0, 20] = 1
	L_zero[0, [2, 5, 8, 11, 20, 22]] = 1.0 / 6.0
	#L_zero[0, 22] = 1

	ref_d = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x, L_zero, ref_d))[0]


	mut_map = numpy.zeros((4, len(ref_seq)))


	for j in range(0, len(ref_seq)) :

		if ref_seq[j] == 'X' :
			continue

		for base in [0, 1, 2, 3] :

			var_x = numpy.zeros(ref_x.shape)
			var_x[:, :, :] = ref_x[:, :, :]

			var_x[0, j, :] = 0
			var_x[0, j, base] = 1

			y_var_hat = logit(cnn.get_online_prediction(var_x, L_zero, ref_d))[0]

			mut_map[3-base, j] = y_var_hat - y_ref_hat

	mut_map[:, 49:49+6] *= downscale_factor

	ref_seq_small_length = 0
	ref_seq_small = ''
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			ref_seq_small_length += 1
			ref_seq_small += ref_seq[j]

	mut_map_small = numpy.zeros((4, ref_seq_small_length))
	k = 0
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			mut_map_small[:, k] = mut_map[:, j]
			k +=1


	ref_seq = ref_seq_small
	mut_map = mut_map_small


	'''f = plt.figure(figsize=(48, 3))

	#mut_map = mut_map - (mut_map.max() + mut_map.min()) / 2.0

	plt.pcolor(mut_map,cmap=cm.RdBu_r,vmin=-numpy.abs(mut_map).max(), vmax=numpy.abs(mut_map).max())
	#plt.pcolor(mut_map,cmap=cm.RdBu_r,vmin=mut_map.min(), vmax=mut_map.max())

	plt.colorbar()
	#plt.xlabel('Sequence position')
	#plt.title('Mutation map of ' + name)


	ref_seq_list = []
	for c in ref_seq :
		ref_seq_list.append(c)

	plt.xticks(numpy.arange(len(ref_seq)) + 0.5, ref_seq_list)
	
	plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'])#BASEPAIR TO INDEX FLIPPED ON PURPOSE TO COUNTER CONVOLVE

	plt.axis([0, mut_map.shape[1], 0, 4])

	plt.gca().xaxis.tick_top()

	#plt.savefig('mut_one_' + name + ".png")
	plt.show()
	plt.close()'''




	fig, ax = plt.subplots(2, 1, figsize=(36, 3))


	#fig, ax = plt.subplots(figsize=(10,3))

	bias = numpy.max(numpy.sum(mut_map[:, :], axis=0)) / 3.0 + 0.5
	max_score = numpy.min(numpy.sum(mut_map[:, :], axis=0)) / 3.0 * -1 + bias
	#max_score = numpy.min(numpy.sum(mut_map[:, :], axis=0)) / 3.0 * -1

	
	#for i in range(0, mut_map.shape[1]) :
	for i in range(plot_start, plot_end) :
		mutability_score = numpy.sum(mut_map[:, i]) / 3.0 * -1 + bias
		#mutability_score = numpy.sum(mut_map[:, i]) / 3.0 * -1

		letterAt(ref_seq[i], i + 0.5, 0, mutability_score, ax[0])

	ax[0].plot([0, mut_map.shape[1]], [bias, bias], color='black', linestyle='--')

	plt.sca(ax[0])
	#plt.xticks(range(1,x))
	#plt.xlim((0, mut_map.shape[1])) 
	plt.yticks([0.5, bias, max_score], [round(bias - 0.5, 2), 0, round((max_score - bias) * -1, 2)])
	plt.xlim((plot_start, plot_end)) 
	plt.ylim((0, max_score)) 
	plt.tight_layout()   


	#pcm = ax[0].pcolormesh(X, Y, Z1,
	#                       norm=MidpointNormalize(midpoint=0.),
	#                       cmap='RdBu_r')
	#pcm = ax.pcolor(mut_map, norm=MidpointNormalize(midpoint=0.), cmap='RdBu_r')
	pcm = ax[1].pcolor(mut_map, cmap='RdBu_r', vmin=-numpy.abs(mut_map).max(), vmax=numpy.abs(mut_map).max())

	#fig.colorbar(pcm, ax=ax[0])

	
	plt.sca(ax[1])

	ref_seq_list = []
	for c in ref_seq :
		ref_seq_list.append(c)
	plt.xticks(numpy.arange(len(ref_seq)) + 0.5, ref_seq_list)
	
	plt.yticks([0.5, 1.5, 2.5, 3.5], ['T', 'G', 'C', 'A'])

	plt.gca().xaxis.tick_top()

	#plt.axis([0, mut_map.shape[1], 0, 4])   
	plt.axis([plot_start, plot_end, 0, 4])

	plt.savefig(name + '.svg', bbox_inches='tight')
	plt.savefig(name + '.png', bbox_inches='tight')
	#plt.show()
	plt.close()


	#cnn.get_logo(mut_map, file_path='' + 'mut_one_' + name + "_logo.png", seq_length=mut_map.shape[1], base_seq='')

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

def mut_map_cuts_compact(cnn, ref_seq, name, plot_start, plot_end, n_show) :

	ref_x = numpy.zeros((1, len(ref_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x[0, j, 3] = 1
		else :
			ref_x[0, j, :] = 0.25

	L_zero = numpy.zeros((1, 36))
	ref_d = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x, L_zero, ref_d))[0]

	positions = numpy.arange(0, 186).tolist()
	y_ref_hat_cuts = numpy.ravel(cnn.get_online_prediction_distrib(ref_x, L_zero, ref_d, positions))


	mut_map = numpy.zeros((4, len(ref_seq)))

	y_var_hat_cuts = numpy.zeros((4, len(ref_seq), len(positions)))


	for j in range(0, len(ref_seq)) :

		if ref_seq[j] == 'X' :
			continue

		for base in [0, 1, 2, 3] :

			var_x = numpy.zeros(ref_x.shape)
			var_x[:, :, :] = ref_x[:, :, :]

			var_x[0, j, :] = 0
			var_x[0, j, base] = 1

			y_var_hat = logit(cnn.get_online_prediction(var_x, L_zero, ref_d))[0]

			y_var_hat_cuts[3-base, j, :] = cnn.get_online_prediction_distrib(var_x, L_zero, ref_d, positions)

			mut_map[3-base, j] = y_var_hat - y_ref_hat


	ref_seq_small_length = 0
	ref_seq_small = ''
	ref_seq_small_start = 0
	has_started = False
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			ref_seq_small_length += 1
			ref_seq_small += ref_seq[j]

		if ref_seq[j] != 'X' and has_started == False :
			ref_seq_small_start = j
			has_started = True

	mut_map_small = numpy.zeros((4, ref_seq_small_length))
	y_var_hat_cuts_small = numpy.zeros((4, ref_seq_small_length, ref_seq_small_length))
	y_ref_hat_cuts_small = y_ref_hat_cuts[ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

	k = 0
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			mut_map_small[:, k] = mut_map[:, j]

			y_var_hat_cuts_small[:, k, :] = y_var_hat_cuts[:, j, ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

			k +=1


	ref_seq = ref_seq_small
	mut_map = mut_map_small

	y_var_hat_cuts = y_var_hat_cuts_small
	y_ref_hat_cuts = y_ref_hat_cuts_small


	color_list = [
		'red',
		'deepskyblue',
		'orange',
		'darkgreen',
		'navy',
		'purple',
		'sandybrown',
		'darkcyan',
		'olivedrab',
		'gold',

		'darksalmon',
		'chartreuse',
		'mediumspringgreen',
		'darkorchid',
		'firebrick'
	]


	top_n_pos = n_show
	top_n_neg = n_show

	#fig, ax = plt.subplots(2, 1, figsize=(16, 12))

	fig = plt.figure(figsize=(18, 6)) 
	gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])
	ax = [ax0, ax1]


	max_y_var_hat = 0
	mut_map_orig = numpy.zeros(mut_map.shape)
	mut_map_orig[:, :] = mut_map[:, :]

	for n in range(0, top_n_pos) :
		max_mut_score = 0
		max_mut_i = 0
		max_mut_j = 0

		pas_chosen = 0
		for i in range(0, mut_map.shape[0]) :
			for j in range(0, mut_map.shape[1]) :
				mut_score = mut_map[i, j]
				if mut_score > max_mut_score and (pas_chosen == 0 or j < 49 or j >= 55) :
					max_mut_score = mut_score
					max_mut_i = i
					max_mut_j = j

					if j >= 49 and j < 55 :
						pas_chosen = 1

		#mut_map[max_mut_i, max_mut_j] = 0
		mut_map[:, max_mut_j] = 0

		#for j in range(0, len(ref_seq)) :
		for j in range(plot_start, plot_end) :
			letterAt(ref_seq[j], j + 0.5, 0, 1.0, ax1)#, color='darkgray')
			if j == max_mut_j :
				if max_mut_i == 3 :
					letterAt('A', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[n])
				if max_mut_i == 2 :
					letterAt('C', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[n])
				if max_mut_i == 1 :
					letterAt('G', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[n])
				if max_mut_i == 0 :
					letterAt('T', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[n])

				##letterAt('UP', j + 0.5, 0.5 + max_mut_score, max_mut_score, ax[1])
				#letterAt('UP', j + 0.5, 1.0 + max_mut_score + 0.1, 0.5, ax[1])

		#ax[0].plot(numpy.arange(len(ref_seq)), y_var_hat_cuts[max_mut_i, max_mut_j, :185], color=color_list[n], linestyle='-')
		ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[n], linestyle='-')

		max_y_var_hat = max(max_y_var_hat, numpy.max(y_var_hat_cuts[max_mut_i, max_mut_j, :185]))

	for n in range(0, top_n_neg) :
		max_mut_score = 0
		max_mut_i = 0
		max_mut_j = 0

		pas_chosen = 0
		for i in range(0, mut_map.shape[0]) :
			for j in range(0, mut_map.shape[1]) :
				mut_score = mut_map[i, j] * -1
				if mut_score > max_mut_score and (pas_chosen == 0 or j < 49 or j >= 55) :
					max_mut_score = mut_score
					max_mut_i = i
					max_mut_j = j

					if j >= 49 and j < 55 :
						pas_chosen = 1

		#mut_map[max_mut_i, max_mut_j] = 0
		mut_map[:, max_mut_j] = 0

		#for j in range(0, len(ref_seq)) :
		for j in range(plot_start, plot_end) :
			letterAt(ref_seq[j], j + 0.5, 0, 1.0, ax[1])#, color='darkgray')
			if j == max_mut_j :
				'''if max_mut_i == 3 :
					letterAt('A', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 2 :
					letterAt('C', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 1 :
					letterAt('G', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 0 :
					letterAt('T', j + 0.5, 1.0, max_mut_score, ax[1], color=color_list[top_n_pos + n])'''

				if max_mut_i == 3 :
					letterAt('A', j + 0.5, -max_mut_score, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 2 :
					letterAt('C', j + 0.5, -max_mut_score, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 1 :
					letterAt('G', j + 0.5, -max_mut_score, max_mut_score, ax[1], color=color_list[top_n_pos + n])
				if max_mut_i == 0 :
					letterAt('T', j + 0.5, -max_mut_score, max_mut_score, ax[1], color=color_list[top_n_pos + n])

				##letterAt('DN', j + 0.5, 0.5 + max_mut_score, max_mut_score, ax[1])
				#letterAt('DN', j + 0.5, 1.0 + max_mut_score + 0.1, 0.5, ax[1])

		ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[top_n_pos + n], linestyle='-')

		max_y_var_hat = max(max_y_var_hat, numpy.max(y_var_hat_cuts[max_mut_i, max_mut_j, :185]))



	#global_max_mut_score = numpy.max(numpy.abs(mut_map_orig))
	global_max_mut_score = numpy.max(mut_map_orig)
	global_min_mut_score = numpy.min(mut_map_orig)


	plt.sca(ax[1])
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))

	#plt.ylim((0, global_max_mut_score + 1.0 + 0.5 + 0.1))
	plt.ylim((global_min_mut_score, global_max_mut_score + 1.0))

	#ax[0].plot(numpy.arange(len(ref_seq)), y_ref_hat_cuts[:185], color='black', linestyle='-', linewidth=3, alpha=0.5)
	ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_ref_hat_cuts[plot_start:plot_end], color='black', linestyle='-', linewidth=3, alpha=0.5)

	ax[0].plot([55, 55], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)
	ax[0].plot([85, 85], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)

	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off')



	plt.sca(ax[0])
	#plt.xticks(range(1,x))
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))

	plt.ylim((0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)))



	plt.tight_layout()
	

	plt.savefig(name + '.svg', bbox_inches='tight')
	plt.savefig(name + '.png', bbox_inches='tight')
	#plt.show()
	plt.close()


def mut_map_cuts_compact_choice_cutlogodds(cnn, ref_seq, var_seq_choices, name, plot_start, plot_end) :

	var_choices = []

	for var_seq_choice in var_seq_choices :
		for j in range(0, len(ref_seq)) :
			if var_seq_choice[j] != ref_seq[j] :
				base = 0
				if var_seq_choice[j] == 'A' :
					base = 3
				elif var_seq_choice[j] == 'C' :
					base = 2
				elif var_seq_choice[j] == 'G' :
					base = 1
				elif var_seq_choice[j] == 'T' :
					base = 0
				var_choices.append([base, j])


	ref_x = numpy.zeros((1, len(ref_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x[0, j, 3] = 1
		else :
			ref_x[0, j, :] = 0.25

	#L_zero = numpy.zeros((1, 36))
	#L_zero[0, 22] = 1
	#ref_d = numpy.zeros((1, 1))

	L_zero = numpy.zeros((1, 36))
	#L_zero[0, 22] = 1
	ref_d = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x, L_zero, ref_d))[0]

	positions = numpy.arange(0, 186).tolist()
	y_ref_hat_cuts = numpy.ravel(cnn.get_online_prediction_distrib(ref_x, L_zero, ref_d, positions))


	mut_map = numpy.zeros((4, len(ref_seq)))

	y_var_hat_cuts = numpy.zeros((4, len(ref_seq), len(positions)))


	for j in range(0, len(ref_seq)) :

		if ref_seq[j] == 'X' :
			continue

		for base in [0, 1, 2, 3] :

			var_x = numpy.zeros(ref_x.shape)
			var_x[:, :, :] = ref_x[:, :, :]

			var_x[0, j, :] = 0
			var_x[0, j, base] = 1

			y_var_hat = logit(cnn.get_online_prediction(var_x, L_zero, ref_d))[0]

			y_var_hat_cuts[3-base, j, :] = cnn.get_online_prediction_distrib(var_x, L_zero, ref_d, positions)

			mut_map[3-base, j] = y_var_hat - y_ref_hat


	ref_seq_small_length = 0
	ref_seq_small = ''
	ref_seq_small_start = 0
	has_started = False
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			ref_seq_small_length += 1
			ref_seq_small += ref_seq[j]

		if ref_seq[j] != 'X' and has_started == False :
			ref_seq_small_start = j
			has_started = True

	'''mut_map_small = numpy.zeros((4, ref_seq_small_length))
	y_var_hat_cuts_small = numpy.zeros((4, ref_seq_small_length, ref_seq_small_length))
	y_ref_hat_cuts_small = y_ref_hat_cuts[ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

	k = 0
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			mut_map_small[:, k] = mut_map[:, j]

			y_var_hat_cuts_small[:, k, :] = y_var_hat_cuts[:, j, ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

			k +=1


	ref_seq = ref_seq_small
	mut_map = mut_map_small

	y_var_hat_cuts = y_var_hat_cuts_small
	y_ref_hat_cuts = y_ref_hat_cuts_small'''


	color_list = [
		'red',
		'deepskyblue',
		'orange',
		'darkgreen',
		'navy',
		'purple',
		'sandybrown',
		'darkcyan',
		'olivedrab',
		'gold',

		'darksalmon',
		'chartreuse',
		'mediumspringgreen',
		'darkorchid',
		'firebrick'
	]


	top_n = len(var_choices)

	#fig, ax = plt.subplots(2, 1, figsize=(16, 12))

	fig = plt.figure(figsize=(18, 6)) 
	gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])
	ax = [ax0, ax1]


	max_y_var_hat = 0
	mut_map_orig = numpy.zeros(mut_map.shape)
	mut_map_orig[:, :] = mut_map[:, :]

	for n in range(0, top_n) :
		max_mut_score = mut_map[var_choices[n][0], var_choices[n][1]]
		max_mut_i = var_choices[n][0]
		max_mut_j = var_choices[n][1]
		max_mut_score_abs = max(numpy.abs(max_mut_score), 0.7)

		#for j in range(0, len(ref_seq)) :
		for j in range(plot_start, plot_end) :
			if j < len(ref_seq) :
				letterAt(ref_seq[j], j + 0.5, 0, 1.0, ax1)#, color='darkgray')
				if j == max_mut_j :
					if max_mut_score > 0 :
						if max_mut_i == 3 :
							letterAt('A', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 2 :
							letterAt('C', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 1 :
							letterAt('G', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 0 :
							letterAt('T', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
					else :
						if max_mut_i == 3 :
							letterAt('A', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 2 :
							letterAt('C', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 1 :
							letterAt('G', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 0 :
							letterAt('T', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])

					#letterAt('UP', j + 0.5, 0.5 + max_mut_score, max_mut_score, ax[1])
					#if max_mut_score > 0 :
					#	letterAt('UP', j + 0.5, 1.0 + max_mut_score_abs + 0.1, 0.5, ax[1])
					#else :
					#	letterAt('DN', j + 0.5, 1.0 + max_mut_score_abs + 0.1, 0.5, ax[1])

		#ax[0].plot(numpy.arange(len(ref_seq)), y_var_hat_cuts[max_mut_i, max_mut_j, :185], color=color_list[n], linestyle='-')
		

		#ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[n], linestyle='-')
		
		y_var_hat_cuts = safe_log(y_var_hat_cuts / (1.0 - y_var_hat_cuts)) - safe_log(y_ref_hat_cuts / (1.0 - y_ref_hat_cuts))

		print('debug1')
		print(y_var_hat_cuts[0, 55:80])
		print(numpy.sum(y_var_hat_cuts[0, 55:80]))

		ax[0].bar(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[n], alpha=0.4)

		max_y_var_hat = max(max_y_var_hat, numpy.max(y_var_hat_cuts[max_mut_i, max_mut_j, :185]))



	#global_max_mut_score = numpy.max(numpy.abs(mut_map_orig))
	global_max_mut_score = numpy.max(mut_map_orig)
	global_min_mut_score = numpy.min(mut_map_orig)


	plt.sca(ax[1])
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))
	#plt.ylim((0, global_max_mut_score + 1.0 + 0.5 + 0.1))
	plt.ylim((global_min_mut_score, global_max_mut_score + 1.0))

	#ax[0].plot(numpy.arange(len(ref_seq)), y_ref_hat_cuts[:185], color='black', linestyle='-', linewidth=2)
	
	#ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_ref_hat_cuts[plot_start:plot_end], color='black', linestyle='-', linewidth=3, alpha=0.7)
	
	#y_ref_hat_cuts = safe_log(y_ref_hat_cuts / (1.0 - y_ref_hat_cuts))

	#ax[0].bar(numpy.arange(plot_end)[plot_start:plot_end], y_ref_hat_cuts[plot_start:plot_end], color='black', alpha=0.4)

	ax[0].plot([55, 55], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)
	ax[0].plot([80, 80], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)

	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off')



	plt.sca(ax[0])
	#plt.xticks(range(1,x))
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))
	#plt.ylim((0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)))



	plt.tight_layout()
	

	plt.savefig(name + '.svg', bbox_inches='tight')
	plt.savefig(name + '.png', bbox_inches='tight')
	#plt.show()
	plt.close()



	#cnn.get_logo(mut_map, file_path='' + 'mut_one_' + name + "_logo.png", seq_length=mut_map.shape[1], base_seq='')


def mut_map_cuts_compact_choice(cnn, ref_seq, var_seq_choices, name, plot_start, plot_end) :

	var_choices = []

	for var_seq_choice in var_seq_choices :
		for j in range(0, len(ref_seq)) :
			if var_seq_choice[j] != ref_seq[j] :
				base = 0
				if var_seq_choice[j] == 'A' :
					base = 3
				elif var_seq_choice[j] == 'C' :
					base = 2
				elif var_seq_choice[j] == 'G' :
					base = 1
				elif var_seq_choice[j] == 'T' :
					base = 0
				var_choices.append([base, j])


	ref_x = numpy.zeros((1, len(ref_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x[0, j, 3] = 1
		else :
			ref_x[0, j, :] = 0.25

	#L_zero = numpy.zeros((1, 36))
	#L_zero[0, 22] = 1
	#ref_d = numpy.zeros((1, 1))

	L_zero = numpy.zeros((1, 36))
	#L_zero[0, 8] = 1
	#L_zero[0, 20] = 1
	L_zero[0, [2, 5, 8, 11, 20, 22]] = 1.0 / 6.0
	#L_zero[0, 22] = 1

	ref_d = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x, L_zero, ref_d))[0]

	positions = numpy.arange(0, 186).tolist()
	y_ref_hat_cuts = numpy.ravel(cnn.get_online_prediction_distrib(ref_x, L_zero, ref_d, positions))


	mut_map = numpy.zeros((4, len(ref_seq)))

	y_var_hat_cuts = numpy.zeros((4, len(ref_seq), len(positions)))


	for j in range(0, len(ref_seq)) :

		if ref_seq[j] == 'X' :
			continue

		for base in [0, 1, 2, 3] :

			var_x = numpy.zeros(ref_x.shape)
			var_x[:, :, :] = ref_x[:, :, :]

			var_x[0, j, :] = 0
			var_x[0, j, base] = 1

			y_var_hat = logit(cnn.get_online_prediction(var_x, L_zero, ref_d))[0]

			y_var_hat_cuts[3-base, j, :] = cnn.get_online_prediction_distrib(var_x, L_zero, ref_d, positions)

			mut_map[3-base, j] = y_var_hat - y_ref_hat


	ref_seq_small_length = 0
	ref_seq_small = ''
	ref_seq_small_start = 0
	has_started = False
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			ref_seq_small_length += 1
			ref_seq_small += ref_seq[j]

		if ref_seq[j] != 'X' and has_started == False :
			ref_seq_small_start = j
			has_started = True

	'''mut_map_small = numpy.zeros((4, ref_seq_small_length))
	y_var_hat_cuts_small = numpy.zeros((4, ref_seq_small_length, ref_seq_small_length))
	y_ref_hat_cuts_small = y_ref_hat_cuts[ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

	k = 0
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] != 'X' :
			mut_map_small[:, k] = mut_map[:, j]

			y_var_hat_cuts_small[:, k, :] = y_var_hat_cuts[:, j, ref_seq_small_start:ref_seq_small_start + ref_seq_small_length]

			k +=1


	ref_seq = ref_seq_small
	mut_map = mut_map_small

	y_var_hat_cuts = y_var_hat_cuts_small
	y_ref_hat_cuts = y_ref_hat_cuts_small'''


	color_list = [
		'red',
		'deepskyblue',
		'orange',
		'darkgreen',
		'navy',
		'purple',
		'sandybrown',
		'darkcyan',
		'olivedrab',
		'gold',

		'darksalmon',
		'chartreuse',
		'mediumspringgreen',
		'darkorchid',
		'firebrick'
	]


	top_n = len(var_choices)

	#fig, ax = plt.subplots(2, 1, figsize=(16, 12))

	fig = plt.figure(figsize=(18, 6)) 
	gs = gridspec.GridSpec(2, 1, height_ratios=[5, 1])

	ax0 = plt.subplot(gs[0])
	ax1 = plt.subplot(gs[1])
	ax = [ax0, ax1]


	max_y_var_hat = 0
	mut_map_orig = numpy.zeros(mut_map.shape)
	mut_map_orig[:, :] = mut_map[:, :]

	for n in range(0, top_n) :
		max_mut_score = mut_map[var_choices[n][0], var_choices[n][1]]
		max_mut_i = var_choices[n][0]
		max_mut_j = var_choices[n][1]
		max_mut_score_abs = max(numpy.abs(max_mut_score), 0.7)

		#for j in range(0, len(ref_seq)) :
		for j in range(plot_start, plot_end) :
			if j < len(ref_seq) :
				letterAt(ref_seq[j], j + 0.5, 0, 1.0, ax1)#, color='darkgray')
				if j == max_mut_j :
					if max_mut_score > 0 :
						if max_mut_i == 3 :
							letterAt('A', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 2 :
							letterAt('C', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 1 :
							letterAt('G', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 0 :
							letterAt('T', j + 0.5, 1.0, max_mut_score_abs, ax[1], color=color_list[n])
					else :
						if max_mut_i == 3 :
							letterAt('A', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 2 :
							letterAt('C', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 1 :
							letterAt('G', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])
						if max_mut_i == 0 :
							letterAt('T', j + 0.5, -max_mut_score_abs, max_mut_score_abs, ax[1], color=color_list[n])

					#letterAt('UP', j + 0.5, 0.5 + max_mut_score, max_mut_score, ax[1])
					#if max_mut_score > 0 :
					#	letterAt('UP', j + 0.5, 1.0 + max_mut_score_abs + 0.1, 0.5, ax[1])
					#else :
					#	letterAt('DN', j + 0.5, 1.0 + max_mut_score_abs + 0.1, 0.5, ax[1])

		#ax[0].plot(numpy.arange(len(ref_seq)), y_var_hat_cuts[max_mut_i, max_mut_j, :185], color=color_list[n], linestyle='-')
		

		ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[n], linestyle='-')
		#ax[0].bar(numpy.arange(plot_end)[plot_start:plot_end], y_var_hat_cuts[max_mut_i, max_mut_j, plot_start:plot_end], color=color_list[n], alpha=0.4)

		max_y_var_hat = max(max_y_var_hat, numpy.max(y_var_hat_cuts[max_mut_i, max_mut_j, :185]))



	#global_max_mut_score = numpy.max(numpy.abs(mut_map_orig))
	global_max_mut_score = numpy.max(mut_map_orig)
	global_min_mut_score = numpy.min(mut_map_orig)


	plt.sca(ax[1])
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))
	#plt.ylim((0, global_max_mut_score + 1.0 + 0.5 + 0.1))
	plt.ylim((global_min_mut_score, global_max_mut_score + 1.0))

	#ax[0].plot(numpy.arange(len(ref_seq)), y_ref_hat_cuts[:185], color='black', linestyle='-', linewidth=2)
	
	ax[0].plot(numpy.arange(plot_end)[plot_start:plot_end], y_ref_hat_cuts[plot_start:plot_end], color='black', linestyle='-', linewidth=3, alpha=0.7)
	#ax[0].bar(numpy.arange(plot_end)[plot_start:plot_end], y_ref_hat_cuts[plot_start:plot_end], color='black', alpha=0.4)

	ax[0].plot([55, 55], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)
	ax[0].plot([80, 80], [0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)], color='green', linestyle='--', linewidth=3)

	plt.tick_params(
		axis='x',          # changes apply to the x-axis
		which='both',      # both major and minor ticks are affected
		bottom='off',      # ticks along the bottom edge are off
		top='off',         # ticks along the top edge are off
		labelbottom='off')



	plt.sca(ax[0])
	#plt.xticks(range(1,x))
	#plt.xlim((0, mut_map.shape[1]))
	plt.xlim((plot_start, plot_end))
	#plt.ylim((0, max(numpy.max(y_ref_hat_cuts[:185]), max_y_var_hat)))



	plt.tight_layout()
	

	plt.savefig(name + '.svg', bbox_inches='tight')
	plt.savefig(name + '.png', bbox_inches='tight')
	#plt.show()
	plt.close()



	#cnn.get_logo(mut_map, file_path='' + 'mut_one_' + name + "_logo.png", seq_length=mut_map.shape[1], base_seq='')

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return numpy.ma.masked_array(numpy.interp(value, x, y))

def hamming_distance(seq1, seq2) :
	dist = 0
	for i in range(0, len(seq1)) :
		if seq1[i] != seq2[i] :
			dist += 1

	return dist

def evaluate_cnn(dataset='general'):

	#dataset = 'snp_general_GENERAL_CI45_SMALL_global'#_ALIGNED

	#dataset = 'snp_general_TOMM5_CI25_SMALL_global'#_ALIGNED
	dataset = 'geuv'#_ALIGNED

	input_datasets = load_input_data(dataset)
	
	ref_x = input_datasets[0]
	var_x = input_datasets[1]
	L_zero = input_datasets[2]

	apadist = input_datasets[3]
	snptype = input_datasets[4]

	output_datasets = load_output_data(dataset)
	
	ref_y = output_datasets[0]
	var_y = output_datasets[1]

	ref_y_spoof = theano.shared(sp.csr_matrix(numpy.zeros((ref_y.eval().shape[0], 186))), borrow=True)
	var_y_spoof = theano.shared(sp.csr_matrix(numpy.zeros((var_y.eval().shape[0], 186))), borrow=True)
	

	ref_d = numpy.ones((ref_y.eval().shape[0], 1))
	ref_d = theano.shared(numpy.asarray(ref_d, dtype=theano.config.floatX), borrow=True)

	var_d = numpy.ones((var_y.eval().shape[0], 1))
	var_d = theano.shared(numpy.asarray(var_d, dtype=theano.config.floatX), borrow=True)


	batch_size = 1

	#run_name = '_Global_Onesided_DoubleDope_Simple'
	#run_name = '_Global_Onesided_DoubleDope_TOMM5_CI25'
	#run_name = '_Global_Onesided_DoubleDope_Simple_TOMM5'
	run_name = '_Global_Onesided2Cuts2_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_TOMM5CI65_alignedonmaxscore'
	#run_name = '_Global_Onesided2_DoubleDope_Simple_TOMM5_APA_Six_30_31_34_GENERALCI45_alignedonmaxscore'

	
	cnn = CutCNN(
		(ref_x, ref_y_spoof, L_zero, ref_d),
		(ref_x, ref_y_spoof, L_zero, ref_d),
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
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34'
		
		#pas_aligned
		dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned',
		store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned'
		
		#wt_dropout
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channeldep',#(_03) _05
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channeldep'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channelindep_05',#(_03) _05
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channelindep_05'

		#dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2smallcuts_finetuned_TOMM5_APA_Six_30_31_34'
	)
	
	print('Trained sublib bias terms:')
	lrW = cnn.output_layer.W.eval()
	lrW = numpy.ravel(lrW[lrW.shape[0] - 36:, 1])
	for i in range(0, len(lrW)) :
		if lrW[i] != 0 :
			print(str(i) + ": " + str(lrW[i]))


	'''cnn.set_data(ref_x, ref_y_spoof, L_zero, ref_d)
	aligned_ref_seqs, aligned_var_seqs, diff_hat, diff = align_predict_snps(cnn, ref_x, ref_y, ref_d, var_x, var_y, L_zero, var_d, apadist, snptype, run_name)

	print('' + 1)'''

	#Generate GEUV SNV mutation maps
	cnn = CutCNN(
		(ref_x, ref_y_spoof, L_zero, ref_d),
		(ref_x, ref_y_spoof, L_zero, ref_d),
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
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34',
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34'
		
		#pas_aligned
		dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned',
		store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned'
		
		#wt_dropout
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channeldep',#(_03) _05
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channeldep'
		#dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channelindep',#(_03) _05
		#store_as_dataset='general' + 'apa_sparse_general' + '_global_onesided2cuts2antimisprimeorigwtdropout_finetuned_TOMM5_APA_Six_30_31_34_pasaligned_channelindep'
	)
	cnn.set_data(ref_x, ref_y_spoof, L_zero, ref_d)
	'''for i in range(0, len(aligned_ref_seqs)) :
		print('Generating mutation maps for variant ' + str(i) + '.')

		ref_seq = aligned_ref_seqs[i][75+1:260+1]
		var_seq = aligned_var_seqs[i][75+1:260+1]

		if hamming_distance(ref_seq, var_seq) <= 1 :
			mut_map(cnn, ref_seq, 'geuv_snp_plots/GEUV_seq_' + str(i), 0, 104)
			mut_map_cuts_compact(cnn, ref_seq, 'geuv_snp_plots/GEUV_seq_' + str(i) + '_Cuts', 0, 104, 3)

			diff_hat_str = str(round(diff_hat[i], 2))
			diff_str = str(round(diff[i], 2))
			mut_map_cuts_compact_choice(cnn, ref_seq, [var_seq], 'geuv_snp_plots/GEUV_seq_' + str(i) + '_ObservedCut_Obs_' + diff_str + '_Pred_' + diff_hat_str, 0, 104)
	'''

	#pasalign_predict_snps(cnn, ref_x, ref_y, ref_d, var_x, var_y, L_zero, var_d, apadist)
	
	#Mutation map
	#http://genome.ucsc.edu/cgi-bin/das/hg19/dna?segment=chr11:5246678,5246777
	#ref_seq = 'XXXXXXXXXXXXXXXXXXXXXXXTAATTTAAATACATCATTGCAATGAAAATAAATGTTTTTTATTAGGCAGAATCCAGATGCTCAAGGCCCTTCATAATATCCCCCAGTAGTTGGACXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

	#http://genome.ucsc.edu/cgi-bin/das/hg19/dna?segment=chr11:5246655,5246839
	#ref_seq = 'ctttttagtaaaatattcagaaataatttaaatacatcattgcaatgaaaataaatgttttttattaggcagaatccagatgctcaaggcccttcataatatcccccagtttagtagttggacttagggaacaaaggaacctttaatagaaattggacagcaagaaagcgagcttagtgatactt'.upper()
	#ref_seq = 'ctttttagtaaaatattcagaaataatttaaatacatcattgcaatgaaaataaatgttttttattaggcagaatccagatgctcaaggcccttcataatatccxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'.upper()


	#ref_seq = 'TCAACAACTACTACCAGGACGCCTGAGGCTGCTCTGCGGGGGACGTGCACAATAAAGGTGTTCTCGGACCCTGTCTGCAGCTCTCTCTCCTGGGTGGGGGCTGAGGACAGGGCAGGGTGGGGGAACAATGACCACAGAGACAGACTCGAGGGAGTTGCTGGTTTATTTAGGAGCCTGGGTCCAGC'
	#ref_seq = 'CTGGGCCCAGAGTCCAGCCGTCCAGCTGCTCCTTTCCCAGTTTGATTTCAATAAATCCGTCAAGTACCATACGATCCCA'
	#ref_seq = ref_seq + ('X' * (185 - len(ref_seq)))

	#ref_seq = 'TGTATCGTCAGGTGGGATAATCCTTACCTGTTCCTCCTCCGGAGGGCAGATTAGAACATGATGATTGGAGATGCATGAAACGTGATTAACGTCTCTGCGTAATCAGGACTTGCAACACCCTGATTGCTCCTGTCTGATTCTTTCTGACGATCACTTACATTTGTGTTATGCTGATTAGCAGATAT'
	#ref_seq = 'TCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATAAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCATGGGGCTGGGGAGGGAGAACTGC'
	#var_seq = 'TCCCAACGGGCCCTCCTCCCCTCCTTGCACCGGCCCTTCCTGGTCTTTGAATGAAGTCTGAGTGGGCAGCAGCCTGTGTGTGCCTGGGTTCTCTCTATCCCGGAATGTGCCAACAATGGAGGTGTTTACCTGTCTCAGACCAAGGACCTCTCTGCAGCTGCATGGGGCTGGGGAGGGAGAACTGC'

	#ref_seq = 'CTCCTTCTCCCTTTTTATATCCCATTTTTATATCGATCTCTTATTTTACAATAAAACTTTGCTGCCACCTGTGTGTCTGAGGGGTGAACGCCAGTGCAGGCTACTGGGGTCAGCAGGTGCAGGGGTGAGTGAGGAGGTGCTGGGAAGCAGCCACCTGAGTCTGCAATGAGTGTGGGCTGGGGGGC'
	#var_seq = 'CTCCTTCTCCCTTTTTATATCCCATTTTTATATCGATCTCTTATTTTACAATAAAACTTTGCTGCCACCTGTGTGTCTGAGTGGTGAACGCCAGTGCAGGCTACTGGGGTCAGCAGGTGCAGGGGTGAGTGAGGAGGTGCTGGGAAGCAGCCACCTGAGTCTGCAATGAGTGTGGGCTGGGGGGC'
	
	#ref_seq = 'CTCCTTCTCCCTTTTTATATCCCATTTTTATATCGATCTCTTATTTTACAATAAAACTTTGCTGCCACCTGTGTGTCTGAGGGGTGAACGCCAGTGCAGGCTACTGGGGTCAGCAGGTGCAGGGGTGAGTGAGGAGGTGCTGGGAAGCAGCCACCTGAGTCTGCAATGAGTGTGGGCTGGGGGGC'
	#var_seq1 = 'CTCCTTCTCCCTTTTTATATCCCATTTTTATATCGATCTCTTATTTTACAATAAAACTTTGCTGCCTCCTGTGTGTCTGAGGGGTGAACGCCAGTGCAGGCTACTGGGGTCAGCAGGTGCAGGGGTGAGTGAGGAGGTGCTGGGAAGCAGCCACCTGAGTCTGCAATGAGTGTGGGCTGGGGGGC'
	
	#mut_map_cuts_compact_choice(cnn, ref_seq, [var_seq1], 'TP53CutSiteDeletion1', 0, 186)

	#ref_seq = 'ACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCATTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTAGTTCAAACCTTGGGAAAATACA'
	#var_seq1 = 'ACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACATTTATTTTCGTTGCAATGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTAGTTCAAACCTTGGGAAAATACA'
	
	#ref_seq = 'ACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACGGGGGGGGGCGGGGCGTTGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTAGTTCAAACCTTGGGAAAATACA'
	#var_seq1 = 'ACTAAACTGGGGGATATTATGAAGGGCCTTGAGCATCTGGATTCTGCCTAATAAAAAACGGGGGGGGGCGGGGCGTTGATGTATTTAAATTATTTCTGAATATTTTACTAAAAAGGGAATGTGGGAGGTCAGTGCATTTAAAACATAAAGAAATGAAGAGCTAGTTCAAACCTTGGGAAAATACA'

	#mut_map_cuts_compact_choice(cnn, ref_seq, [var_seq1], 'HBBCutSiteDeletion1', 0, 186)

	#ref_seq = '............CATTACTCGCATCCAAATGTTTTTGGCTTCTCTCTGGAATAAAAAAGCTACTTATTTCAATGATGATGTGCTTTCCCGCATATTAGCAGCCAATTAAGCCTCAACGTACAGGTTATCCTGACGGAATAAAATTAACAATGACCTCGGCGTCGGTATTCTAAATGCGCCGTCCT'
	#var_seq = '............CATTACTCGCATCCAAATGTTTTTGGCTTCTCTCTGGAACAAAAAAGCTACTTATTTCAATGATGATGTGCTTTCCCGCATATTAGCAGCCAATTAAGCCTCAACGTACAGGTTATCCTGACGGAATAAAATTAACAATGACCTCGGCGTCGGTATTCTAAATGCGCCGTCCT'

	#ref_seq = '..........CATTACTCGCATCCAGTAGCTAGCTGTTCGATTAGTTTCAATAAAATTTGGGCTAAGGCACTGGGTATGTGAATTCAAGAAGTGGTCAGCCAATTAAGCCCGCGTGCTAATAGTGCCGTGCAAATAAAAAATTACATCTCTGTTGAGCTTCACCTTTATGCGTTCCTTTAGCTAC'
	#var_seq = '..........CATTACTCGCATCCAGTAGCTAGCTGTTCGATTAGTTTCAATGAAATTTGGGCTAAGGCACTGGGTATGTGAATTCAAGAAGTGGTCAGCCAATTAAGCCCGCGTGCTAATAGTGCCGTGCAAATAAAAAATTACATCTCTGTTGAGCTTCACCTTTATGCGTTCCTTTAGCTAC'
	#var_seq = '..........CATTACTCGCATCCAGTAGCTAGCTGTTCGATTAGTTTCGACAAAATTTGGGCTAAGGCACTGGGTATGTGAATTCAAGAAGTGGTCAGCCAATTAAGCCCGCGTGCTAATAGTGCCGTGCAAATAAAAAATTACATCTCTGTTGAGCTTCACCTTTATGCGTTCCTTTAGCTAC'



	#ref_seq = 'CTGGGCCCAGAGTCCAGCCGTCCAGCTGCTCCTTTCCCAGTTTGATTTCAATAAATCGATGAATAAGAGTCCTGACAAGGGTGCCAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

	print('Test 11')
	ref_seq = 'ACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCGAGCCTCAATGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCACCTGGTGCACGCTGGTAGTCCG'
	var_seq = 'ACCAATCCCGTGAAAGAATTATTTTTGTGTTTCTAAAACTATGGTTCCCAATAAAAGTGACTCTCAGCAAGCCTCAATGCTCCCAGTGCTATTCATGGGCAGCTCTCTGGGCTCAGGAAGAGCCAGTAATACTACTGGATAAAGAAGACTTAAGAATCCACCACCTGGTGCACGCTGGTAGTCCG'

	mut_map(cnn, ref_seq, 'Tester1DropoutSimulation', downscale_factor=0.5)
	mut_map_cuts_compact_choice(cnn, ref_seq, [var_seq], 'Tester1Dropout', 0, 120)
	mut_map_cuts_compact_choice_cutlogodds(cnn, ref_seq, [var_seq], 'Tester1Dropout_Logodds', 0, 120)

	ref_x = numpy.zeros((1, len(ref_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x[0, j, 3] = 1

	var_x = numpy.zeros((1, len(var_seq), 4))
	for j in range(0, len(var_seq)) :
		if var_seq[j] == 'A' :
			var_x[0, j, 0] = 1
		elif var_seq[j] == 'C' :
			var_x[0, j, 1] = 1
		elif var_seq[j] == 'G' :
			var_x[0, j, 2] = 1
		elif var_seq[j] == 'T' :
			var_x[0, j, 3] = 1

	L_zero = numpy.zeros((1, 36))
	#L_zero[0, 8] = 1
	#L_zero[0, 20] = 1
	L_zero[0, [2, 5, 8, 11, 20, 22]] = 1.0 / 6.0
	#L_zero[0, 22] = 1

	ref_d = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x, L_zero, ref_d))[0]
	y_var_hat = logit(cnn.get_online_prediction(var_x, L_zero, ref_d))[0]

	print(y_ref_hat)
	print(y_var_hat)

	print(y_var_hat - y_ref_hat)

	print(cnn.get_online_prediction(ref_x, L_zero, ref_d)[0])
	print(cnn.get_online_prediction(var_x, L_zero, ref_d)[0])


	positions = numpy.arange(0, 186).tolist()

	y_ref_hat_cuts = numpy.ravel(cnn.get_online_prediction_distrib(ref_x, L_zero, ref_d, positions))
	y_var_hat_cuts = numpy.ravel(cnn.get_online_prediction_distrib(var_x, L_zero, ref_d, positions))


	#ref_seq = 'TGCAGCTTATAATGGTTACAAATAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
	#var_seq = 'TGCAGCTTATAATGGTTACATGTAAAGCAATAGCATCACAAATTTCACAAATAAAGCATTTTTTTCACTGCATTCTAGTTGTGGTTTGTCCAAACTCATCAATGTATCTTAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'


	#ref_seq = 'CTGGGCCCAGAGTCCAGCCGTCCAGCTGCTCCTTTCCCAGTTTGATTTCAATAAATCGATGAATAAGAGTCCTGACAAGGGTGCCAXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'

	#mut_map(cnn, ref_seq, 'CENPBCuts2')
	#mut_map_cuts_compact_choice(cnn, var_seq, [var_seq], 'Tester2', 0, 111)

	#mut_map_cuts_compact(cnn, ref_seq, 'Tester')
	print('' + 1)

	#Specific seq predictor
	'''ref_seq = ('X' * (75 + 1)) + ('X' * 23) + 'TAATTTAAATACATCATTGCAATGAAAATAAATGTTTTTTATTAGGCAGAATCCAGATGCTCAAGGCCCTTCATAATATCCCCCAGTAGTTGGAC'  + ('X' * (106 - 25))
	var_seq = ('X' * (75 + 1)) + ('X' * 23) + 'TAATTTAAATACATCATTGCAATGAAAATAAATGTTTTTTATTAGGCAGAATCCAGATGCTCAAGGCCCTTCATAATATCCCCCAGTAGTTGGAC'  + ('X' * (106 - 25))
	
	#ref_seq = 'GGGGGCAAATTTGGCACCTGCCCCCACTTGGGACTTTGGTCTTGCTGAAAATAAATATTTTTCTTTTTCAAAGACTTTGTGATTCCCCAGATAGGTTGCCTGAAATGGGTAAGAGGATGGAGGACTCAACAGTGCAGGGTTTGAGGCCTGAATGGTCATCTGCATCAXXXXXXXXXXXXXXXXXX'
	#var_seq = 'GGGGGCAAATTTGGCACCTGCCCCCACTTGGGACTTTGGTCTTGCTGXXAATAAATATTTTTCTTTTTCAAAGACTTTGTGATTCCCCAGATAGGTTGCCTGAAATGGGTAAGAGGATGGAGGACTCAACAGTGCAGGGTTTGAGGCCTGAATGGTCATCTGCATCAXXXXXXXXXXXXXXXXXX'

	#ref_seq = 'GGGGGCAAATTTGGCACCTGCCCCCACTTGGGACTTTGGXXXXXXXXXXAATAAATTTTTTTCTTTTTCAAAGACTTTGTGATTCCCCAGATAGGTTGCCTGAAATGGGTAAGAGGATGGAGGACTCAACAGTGCAGGGTTTGAGGCCTGAATGGTCATCTGCATCAXXXXXXXXXXXXXXXXXX'
	#var_seq = 'GGGGGCAAATTTGGCACCTGCCCCCACTTGGGACTTTGGXXXXXXXXAAAATAAATTTTTTTCTTTTTCAAAGACTTTGTGATTCCCCAGATAGGTTGCCTGAAATGGGTAAGAGGATGGAGGACTCAACAGTGCAGGGTTTGAGGCCTGAATGGTCATCTGCATCAXXXXXXXXXXXXXXXXXX'

	#ref_seq = 'GTGGCTCATTTTCTGGCAAATGGAGGCACGAACGCAGGGGCCAAATAGCAATAAATGGGTTTTGTTTTTTTTTTGCAATAACTTATTGAAGTCAGCAGGGCATCCTTCCCTAGTATGCTTCCTGGGGCGTGTCTAGGGGCCAGCTCCCTTCCCTGGGGGCAGCCCTXXXXXXXXXXXXXXXXXXX'
	#var_seq = 'GTGGCTCATTTTCTGGCAAATGGAGGCACGAACGCAGGGGCCAAATAGCAATAAATGGGTTTTGTTTTTTTTTTGCAGTGACTTATTGAAGTCAGCAGGGCATCCTTCCCTAGTATGCTTCCTGGGGCGTGTCTAGGGGCCAGCTCCCTTCCCTGGGGGCAGCCCTXXXXXXXXXXXXXXXXXXX'

	#ref_seq = 'GGAGCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAAAAAAGTCTCCTGGGAGTGGGAAGGGAGCTAGTGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGAGACAGTTTCGCTCTGTTXXXXXXXXXXXXXXXXX'
	#var_seq = 'GGAGCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAACCAAGTCTCCTGGGAGTGGGAAGGGAGCTAGTGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGAGACAGTTTCGCTCTGTTXXXXXXXXXXXXXXXXX'


	#APADB into DoubleDope
	#ref_seq = 'XXXXXXXXXCATTACTCGCATCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAAAAAAGTCTCCTGGGAGTGGGAAGCCAATTAAGCCATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGAGACAGTTTCGCTCTGTTXXXXXXXXXXXXCTACG'
	#var_seq = 'XXXXXXXXXCATTACTCGCATCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAACCAAGTCTCCTGGGAGTGGGAAGCCAATTAAGCCATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGAGACAGTTTCGCTCTGTTXXXXXXXXXXXXCTACG'

	#APADB into Simple
	#ref_seq = 'ATCTCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAAAAAAGTCTCCTGGGAGTGGGAAGGGAGCTAGTGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGACGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#var_seq = 'ATCTCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAACCAAGTCTCCTGGGAGTGGGAAGGGAGCTAGTGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGACGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'

	#ref_seq = 'ATCTCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAAAAAAGTCTCCTGGGAGTGGGAAGGGATTAAATGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGACGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#var_seq = 'ATCTCTGCTGTGTATAGACTGCCAAATGTGAAGTATTTATATTGTATTCAATAAACTATACTTAAGAGTGTTCAACCAAGTCTCCTGGGAGTGGGAAGGGATTAAATGGATACTCCCTATTTCACAAACTTTTCTTTTTTTTTTTTTTTGACGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	


	#DoubleDope
	#ref_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGAAAAAAAAAGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'
	#var_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGAAAACCAAAGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'

	#ref_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGTTAAAAAATGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'
	#var_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGTTAACCAATGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'

	#ref_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGTAAAAAAAAGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'
	#var_seq = 'XXXXXXXXXXCATTACTCGCATCCATAGTAATAGTGACGGTCGCATTCTAATAAATTCTATGAGAGTAGCGTAAACCAAAGGTTTTTGGGTAGGAACAGCCAATTAAGCCATTCACTCCCATCTTCCTCCCACATATAAATTCTGACCTTAAGCTGATGGCTTACCTTTGGGAAAGAGCTTCTACG'


	#Simple into DoubleDope
	#ref_seq = 'XXXXXXXXXCATTACTCGCATCCAAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCTAAAAAAAATTCCCTTTTTGTGGTGAGCCAATTAAGCCATTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCACTACG'
	#var_seq = 'XXXXXXXXXCATTACTCGCATCCAAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCTAAACCAAATTCCCTTTTTGTGGTGAGCCAATTAAGCCATTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCACTACG'


	#Simple
	#ref_seq = 'ATCTTCTCTGTGTCTGGTACATACAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCACAAAAAATATCCCTTTTTGTGGTGTGAGATTAAAGGGTTTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#var_seq = 'ATCTTCTCTGTGTCTGGTACATACAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCACAACCAATATCCCTTTTTGTGGTGTGAGATTAAAGGGTTTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'


	#ref_seq = 'ATCTTCTCTGTGTCTGGTACATACAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCTAAAAAAAATTCCCTTTTTGTGGTGTGAGATTAAAGGGTTTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'
	#var_seq = 'ATCTTCTCTGTGTCTGGTACATACAACCCTAAGCTGTAAACAGTGGTTCAATAAATTTATTTACTGGCATCTAAACCAAATTCCCTTTTTGTGGTGTGAGATTAAAGGGTTTTACTCTAGGGAGCAGGTCCGTTATGTTTTACTCCCTACGCGCCTAACCCTAAGCAGATTCTTCATGCAATTGT'


	#ref_seq = 'CCCCCGCCCAGACTGCAGGCTCCCCTTCCTGCACCACCATTGTCTCAGCAGTAAAGGCGACATTTGGAACCACAGCATGGCCTTGACCATAGGGGTTCCTCGCAGGCAGAGCCTTGCCTCCTCCTGGGTCCCACCTGGCCTCTGCAGACCTTAGGCTGGGACGGGGXXXXXXXXXXXXXXXXXXX'
	#var_seq = 'CCCCCGCCCAGACTGCAGGCTCCCCTTCCTGCACCACCATTGTCTCAGCAATAAAGGCGACATTTGGAACCACAGCATGGCCTTGACCATAGGGGTTCCTCGCAGGCAGAGCCTTGCCTCCTCCTGGGTCCCACCTGGCCTCTGCAGACCTTAGGCTGGGACGGGGXXXXXXXXXXXXXXXXXXX'



	ref_x_p = numpy.zeros((1, len(ref_seq), 4))
	var_x_p = numpy.zeros((1, len(var_seq), 4))
	for j in range(0, len(ref_seq)) :
		if ref_seq[j] == 'A' :
			ref_x_p[0, j, 0] = 1
		elif ref_seq[j] == 'C' :
			ref_x_p[0, j, 1] = 1
		elif ref_seq[j] == 'G' :
			ref_x_p[0, j, 2] = 1
		elif ref_seq[j] == 'T' :
			ref_x_p[0, j, 3] = 1
		else :
			ref_x_p[0, j, :] = 0.25
	for j in range(0, len(var_seq)) :
		if var_seq[j] == 'A' :
			var_x_p[0, j, 0] = 1
		elif var_seq[j] == 'C' :
			var_x_p[0, j, 1] = 1
		elif var_seq[j] == 'G' :
			var_x_p[0, j, 2] = 1
		elif var_seq[j] == 'T' :
			var_x_p[0, j, 3] = 1
		else :
			var_x_p[0, j, :] = 0.25

	L_zero_p = numpy.zeros((1, 36))
	d_p = numpy.ones((1, 1))

	y_ref_hat = logit(cnn.get_online_prediction(ref_x_p, L_zero_p, d_p))[0]
	y_var_hat = logit(cnn.get_online_prediction(var_x_p, L_zero_p, d_p))[0]

	print('y_ref_hat = ' + str(y_ref_hat))
	print('y_var_hat = ' + str(y_var_hat))'''

	print(1 + '')


	cnn.set_data(ref_x, ref_y, L_zero, ref_d)
	y_ref_hat = logit(cnn.get_prediction())
	y_ref = logit(ref_y.eval()[:,1])

	cnn.set_data(var_x, var_y, L_zero, var_d)
	y_var_hat = logit(cnn.get_prediction())
	y_var = logit(var_y.eval()[:,1])



	#Debug
	print("apadist 3561")
	print('ref hat')
	print(y_ref_hat[apadist == 3561])
	print('var hat')
	print(y_var_hat[apadist == 3561])

	print("apadist 265")
	print('ref hat')
	print(y_ref_hat[apadist == 265])
	print('var hat')
	print(y_var_hat[apadist == 265])


	diff = y_var - y_ref
	diff_hat = y_var_hat - y_ref_hat


	SSE_diff = (diff - diff_hat).T.dot(diff - diff_hat)

	y_diff_average = numpy.average(diff, axis=0)

	SStot_diff = (diff - y_diff_average).T.dot(diff - y_diff_average)

	RMSE_diff = numpy.sqrt(SSE_diff / float(len(y_ref)))

	MAE_diff = numpy.mean(numpy.abs(diff_hat - diff))

	diff_set_dir_accuracy = numpy.count_nonzero(numpy.sign(diff) == numpy.sign(diff_hat))


	print("")
	print("Logodds diff R^2:")
	print(1.0 - (SSE_diff / SStot_diff))
	print("Logodds diff mean abs error:")
	print(MAE_diff)

	print("Logodds diff Classification accuracy: " + str(diff_set_dir_accuracy) + "/" + str(y_ref.shape[0]) + " = " + str(float(diff_set_dir_accuracy) / float(y_ref.shape[0])))

	fig = plt.figure()
	ax1 = fig.add_subplot(111)

	col = ax1.scatter(diff_hat, diff, c = 'red', alpha=1.0)
	ax1.plot([-5, 5], [-5, 5], c='yellow')
	ax1.plot([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1], [0, 0], c='green')
	ax1.plot([0, 0], [numpy.min(diff) * 1.1, numpy.max(diff) * 1.1], c='green')
	ax1.set_xlim([numpy.min(diff_hat) * 1.1, numpy.max(diff_hat) * 1.1])
	ax1.set_ylim([numpy.min(diff) * 1.1, numpy.max(diff) * 1.1])
	ax1.set_xlabel('Predicted Proximal usage logodds diff', fontsize=22)
	ax1.set_ylabel('Target Proximal usage logodds diff', fontsize=22)
	ax1.set_title('GEUV APA SNP Log Diff (R^2 = ' + str(round(1.0 - (SSE_diff / SStot_diff), 2)) + ', Acc = ' + str(diff_set_dir_accuracy) + "/" + str(y_ref.shape[0]) + ')', fontsize=18)
	
	for i in range(0, len(diff)):
		#ax1.annotate(snp_index[i] + 2, (diff_hat[i], diff[i]))
		'''annotation = ''
		if snptype[i] == 1 :
			annotation = 'HET'
		elif snptype[i] == 2 :
			annotation = 'HOM'
		
		if snpregion[i] == 1 :
			annotation += ' UP'
		elif snpregion[i] == 2 :
			annotation += ' PAS'
		elif snpregion[i] == 3 :
			annotation += ' DN'
		'''
		annotation = '(' + str(apadist[i]) + ')'

		ax1.annotate(annotation, (diff_hat[i], diff[i]), size=8)

	#plt.savefig("cnn_snp_logodds_diff_global" + run_name + ".png")
	plt.show()
	plt.close()

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
