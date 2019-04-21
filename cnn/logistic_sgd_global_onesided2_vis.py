
import cPickle
import gzip
import os
import sys
import time

import numpy
import scipy.sparse as sp
import scipy.io as spio

import theano
import theano.tensor as T

class TrainableImage(object):

	def __init__(self, rng, n_images, n_length, seq, start_seqs):
		
		mask = numpy.zeros((n_images, len(seq), 4))
		is_trainable = []
		is_not_trainable = []
		is_zero = []

		if start_seqs is None :
			for i in range(0, len(seq)) :
				if seq[i] == 'A' :
					mask[:, i, 0] = 1
					is_not_trainable.append(i)
				elif seq[i] == 'C' :
					mask[:, i, 1] = 1
					is_not_trainable.append(i)
				elif seq[i] == 'G' :
					mask[:, i, 2] = 1
					is_not_trainable.append(i)
				elif seq[i] == 'T' :
					mask[:, i, 3] = 1
					is_not_trainable.append(i)
				elif seq[i] == 'N' :
					mask[:, i, :] = rng.uniform(low=-0.2, high=0.2, size=(n_images, 4))
					is_trainable.append(i)
				else :
					is_not_trainable.append(i)
					is_zero.append(i)
		else :
			for j in range(0, n_images) :
				start_seq = start_seqs[j]
				for i in range(0, len(start_seq)) :
					if start_seq[i] == '.' :
						is_not_trainable.append(i)
						is_zero.append(i)
					else :
						if start_seq[i] == 'A' :
							mask[j, i, 0] = 1
							is_trainable.append(i)
						elif start_seq[i] == 'C' :
							mask[j, i, 1] = 1
							is_trainable.append(i)
						elif start_seq[i] == 'G' :
							mask[j, i, 2] = 1
							is_trainable.append(i)
						elif start_seq[i] == 'T' :
							mask[j, i, 3] = 1
							is_trainable.append(i)
						elif seq[i] == 'N' :
							mask[j, i, :] = rng.uniform(low=-0.2, high=0.2, size=(4,))
							is_trainable.append(i)


		print(mask[0, :, :])


		#print(is_trainable)
		#print(is_not_trainable)

		W_trainable = theano.shared(
			value=numpy.array(
			mask[:, is_trainable, :],
			dtype=theano.config.floatX
			),
			name='masked_W',
			borrow=True
		)
		W_not_trainable = theano.shared(
			value=numpy.array(
			mask[:, is_not_trainable, :],
			dtype=theano.config.floatX
			),
			name='masked_W',
			borrow=True
		)
		'''W = theano.shared(
			value=numpy.array(
			mask,
			dtype=theano.config.floatX
			),
			name='masked_W',
			borrow=True
		)'''
		
		
		self.is_trainable = is_trainable
		self.is_not_trainable = is_not_trainable
		

		W = theano.shared(
			value=numpy.zeros(
			(n_images, len(seq), 4),
			dtype=theano.config.floatX
			),
			name='masked_W',
			borrow=True
		)
		W = T.set_subtensor(W[:, is_trainable, :], W_trainable)
		#W = T.set_subtensor(W[:, is_not_trainable, :], W_not_trainable)

		self.W = W
		#self.outputs = self.W
		#self.outputs = T.switch(self.W<0, 0, self.W)
		#self.outputs = T.nnet.softmax(self.W) #T.cast(T.switch(self.W >= T.max(self.W, axis=1), 1, 0), 'float64')
		
		outputs = T.nnet.softmax(self.W.reshape((n_images * n_length, 4))).reshape((n_images, n_length, 4))
		
		if start_seqs is None :
			if len(is_zero) > 1 :
				outputs = T.set_subtensor(outputs[:, is_zero, :], 0)
			
			outputs = T.set_subtensor(outputs[:, is_not_trainable, :], W_not_trainable)

		#self.outputs[:, zero_region[0]:zero_region[1], :] = 0

		self.outputs = outputs

		#T.eq(x, x.max())

		
		self.params = [W_trainable]#[self.W[:, :zero_region[0], :], self.W[:, zero_region[1]:, :]]

class TrainableImageSimple(object):

	def __init__(self, rng, n_images, n_length, zero_region=[40,73]):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		# start-snippet-1
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		


		self.W = theano.shared(
			numpy.asarray(
				rng.uniform(low=-0.2, high=0.2, size=(n_images, n_length, 4)),
					dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)

		'''self.W = theano.shared(
			value=numpy.zeros(
				(n_length, 4),
				dtype=theano.config.floatX
			),
			name='W',
			borrow=True
		)'''
		# initialize the baises b as a vector of n_out 0s
		'''self.b = theano.shared(
			value=numpy.zeros(
				(n_out,),
				dtype=theano.config.floatX
			),
			name='b',
			borrow=True
		)'''
		
		#self.outputs = self.W
		#self.outputs = T.switch(self.W<0, 0, self.W)
		#self.outputs = T.nnet.softmax(self.W) #T.cast(T.switch(self.W >= T.max(self.W, axis=1), 1, 0), 'float64')
		self.outputs = T.nnet.softmax(self.W.reshape((n_images * n_length, 4))).reshape((n_images, n_length, 4))

		#self.outputs[:, zero_region[0]:zero_region[1], :] = 0

		#self.outputs = T.concatenate([self.outputs, T.zeros([x.shape[0], y.shape[1]], dtype=theano.config.floatX), self.outputs], axis=1)

		#T.eq(x, x.max())

		self.params = [self.W]#[self.W[:, :zero_region[0], :], self.W[:, zero_region[1]:, :]]


class LogisticRegression(object):
	"""Multi-class Logistic Regression Class

	The logistic regression is fully described by a weight matrix :math:`W`
	and bias vector :math:`b`. Classification is done by projecting data
	points onto a set of hyperplanes, the distance to which is used to
	determine a class membership probability.
	"""
	
	def store_w(self, w_file, W):
		numpy.save(w_file, W)
	
	def store_b(self, b_file, b):
		numpy.save(b_file, b)
	
	def store_model(self, W, b):
		self.store_w(self.w_file, W)
		self.store_b(self.b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)

	def __init__(self, cnn, image, input, n_in, n_out, load_model = False, cost_func='max_score', cost_layer_filter=0, w_file = '', b_file = ''):
		""" Initialize the parameters of the logistic regression

		:type input: theano.tensor.TensorType
		:param input: symbolic variable that describes the input of the
					  architecture (one minibatch)

		:type n_in: int
		:param n_in: number of input units, the dimension of the space in
					 which the datapoints lie

		:type n_out: int
		:param n_out: number of output units, the dimension of the space in
					  which the labels lie

		"""
		# start-snippet-1
		# initialize with 0 the weights W as a matrix of shape (n_in, n_out)
		
		self.cost_func = cost_func

		self.w_file = w_file
		self.b_file = b_file

		self.image = image

		self.cnn = cnn
		self.cost_layer_filter = cost_layer_filter
		
		if load_model == False :
			self.W = theano.shared(
				value=numpy.zeros(
					(n_in, n_out),
					dtype=theano.config.floatX
				),
				name='W',
				borrow=True
			)
			# initialize the baises b as a vector of n_out 0s
			self.b = theano.shared(
				value=numpy.zeros(
					(n_out,),
					dtype=theano.config.floatX
				),
				name='b',
				borrow=True
			)
		else :
			self.W = theano.shared(
				value=self.load_w(w_file + '.npy'),
				name='W',
				borrow=True
			)
			# initialize the baises b as a vector of n_out 0s
			self.b = theano.shared(
				value=self.load_b(b_file + '.npy'),
				name='b',
				borrow=True
			)

		# symbolic expression for computing the matrix of class-membership
		# probabilities
		# Where:
		# W is a matrix where column-k represent the separation hyper plain for
		# class-k
		# x is a matrix where row-j  represents input training sample-j
		# b is a vector where element-k represent the free parameter of hyper
		# plain-k
		self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

		self.s_y_given_x = T.dot(input, self.W) + self.b

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		#self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y, epoch):
		"""Return the mean of the negative log-likelihood of the prediction
		of this model under a given target distribution.

		.. math::

			\frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
			\frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
				\log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
			\ell (\theta=\{W,b\}, \mathcal{D})

		:type y: theano.tensor.TensorType
		:param y: corresponds to a vector that gives for each example the
				  correct label

		Note: we use the mean instead of the sum so that
			  the learning rate is less dependent on the batch size
		"""
		# start-snippet-2
		# y.shape[0] is (symbolically) the number of rows in y, i.e.,
		# number of examples (call it n) in the minibatch
		# T.arange(y.shape[0]) is a symbolic vector which will contain
		# [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
		# Log-Probabilities (call it LP) with one row per example and
		# one column per class LP[T.arange(y.shape[0]),y] is a vector
		# v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
		# LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
		# the mean (across minibatch examples) of the elements in v,
		# i.e., the mean log-likelihood across the minibatch.
		
		
		#return -T.mean(T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])])
		#return -T.sum(T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])])
		#return -T.mean(self.s_y_given_x[:, 1])
		
		'''if self.cost_func == 'max_score' :
			return -T.sum(self.s_y_given_x[:, 1])
		elif self.cost_func == 'target' :
			return -T.sum(T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])])
		else :
			return T.sum(self.s_y_given_x[:, 1])'''

		if self.cost_func == 'max_score_punish_cruns_softer' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005
			return T.switch(T.le(epoch, 4000),#4000#2000#1000
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005,
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.025)
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		elif self.cost_func == 'max_score_punish_cruns_harder' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001#0.0005
			return T.switch(T.le(epoch, 4000),#4000#2000#1000
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001,
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.025)

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		elif self.cost_func == 'max_score_punish_cruns_cstf' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001#0.0005
			return T.switch(T.le(epoch, 4000),#4000#2000#1000
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001 + T.sum(self.image[:, :, 55:-1, 2] * self.image[:, :, 56:, 3]) * 0.5 + T.sum(self.image[:, :, 55:-1, 1] * self.image[:, :, 56:, 3]) * 0.5,
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.025)

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		
		elif self.cost_func == 'max_score_punish_cruns_aruns_cstf' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.1
			return T.switch(T.le(epoch, 4000),#4000#2000#1000
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.5 + T.sum(self.image[:, :, 55:-1, 2] * self.image[:, :, 56:, 3]) * 0.5 + T.sum(self.image[:, :, 55:-1, 1] * self.image[:, :, 56:, 3]) * 0.5,
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.5 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.025)

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		
		elif self.cost_func == 'max_score_punish_cruns_aruns' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.1
			return T.switch(T.le(epoch, 8000),#4000#2000#1000
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.1,
					-T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.0005 + T.sum(self.image[:, :, 55:-1, 0] * self.image[:, :, 56:, 0]) * 0.1 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.025)

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		elif self.cost_func == 'max_score' :
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 0.005 #+ T.sum(self.image[:, :, :, 2]) * 0.0005
			return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :50, 1]) * 0.0005 + T.sum(self.image[:, :, :, 2]) * 0.0005
			
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.1 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.1
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:49, 1] * self.image[:, :, 1:50, 1]) * 0.01 + T.sum(self.image[:, :, 0:49, 2] * self.image[:, :, 1:50, 2]) * 0.005

			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
		
		elif self.cost_func == 'target' :
			return T.sum(T.dot(self.p_y_given_x[:, 1] - y[:,1], self.p_y_given_x[:, 1] - y[:,1])) + T.sum(self.image[:, :, :50, 1]) * 0.1 + T.sum(self.image[:, :, :, 2]) * 0.001 - T.sum( T.sum( self.image * T.log(self.image + 10**-6), axis=1) ) * 0.007 #Entropy term
		



		elif self.cost_func == 'log_target' :
			return T.dot(T.log(y[:,1] / (1 - y[:,1])) - T.log(self.p_y_given_x[:,1] / (1 - self.p_y_given_x[:,1])), T.log(y[:,1] / (1 - y[:,1])) - T.log(self.p_y_given_x[:,1] / (1 - self.p_y_given_x[:,1]))) + T.sum(self.image[:, :, :49, 1]) * 0.05 + T.sum(self.image[:, :, 49:, 1]) * 0.01 #+ T.sum(self.image[:, :, :49, 3]) * 0.005
		
		elif self.cost_func == 'max_layer2_score' :
			return -T.sum(self.cnn.layer1.conv_out[:, self.cost_layer_filter, 0, 0])

		elif self.cost_func == 'max_dense_score' :
			return -T.sum(self.cnn.layer3.output[:, self.cost_layer_filter])

		else :
			return T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 1.0


	def negative_log_likelihood_unsummed(self, y):
		if self.cost_func == 'max_score' :
			return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :45, 1]) * 0.5#0.005
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :-1, 1] * self.image[:, :, 1:, 1]) * 0.005 + T.sum(self.image[:, :, :-1, 2] * self.image[:, :, 1:, 2]) * 0.005
			#return -T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, 0:45, 1] * self.image[:, :, 1:46, 1]) * 0.05 + T.sum(self.image[:, :, 0:45, 2] * self.image[:, :, 1:46, 2]) * 0.05
		elif self.cost_func == 'target' :
			return T.sum(T.dot(self.s_y_given_x[:, 1] - y[:,1], self.s_y_given_x[:, 1] - y[:,1]))
		elif self.cost_func == 'log_target' :
			return T.dot(T.log(y[:,1] / (1 - y[:,1])) - T.log(self.p_y_given_x[:,1] / (1 - self.p_y_given_x[:,1])), T.log(y[:,1] / (1 - y[:,1])) - T.log(self.p_y_given_x[:,1] / (1 - self.p_y_given_x[:,1]))) + T.sum(self.image[:, :, :49, 1]) * 0.05 + T.sum(self.image[:, :, 49:, 1]) * 0.01 #+ T.sum(self.image[:, :, :49, 3]) * 0.005
		
		elif self.cost_func == 'max_layer2_score' :
			return -T.sum(self.cnn.layer1.activation[:, self.cost_layer_filter, :, :])

		elif self.cost_func == 'max_dense_score' :
			return -1 * self.cnn.layer3.output[:, self.cost_layer_filter]

		else :
			return T.sum(self.s_y_given_x[:, 1]) + T.sum(self.image[:, :, :, 1]) * 1.0

	def rsquare(self, y):
		return 1.0 - (T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1]) / T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1])))
	
	def sse(self, y):
		return T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1])
	
	def sst(self, y):
		return T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1]))
	
	def abs_error(self, y):
		return T.mean(abs(y[:,1] - self.p_y_given_x[:,1]))
	
	def recall(self):
		return self.p_y_given_x[:,1]

	def log_loss(self, y):
		return -T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])]


def shared_dataset(data, datatype=theano.config.floatX, borrow=True):
		""" Function that loads the dataset into shared variables"""
		
		shared_data = theano.shared(numpy.asarray(data,
											   dtype=datatype),
								 borrow=borrow)

		return shared_data

if __name__ == '__main__':
	#sgd_optimization_mnist()
	print('cant be main method')