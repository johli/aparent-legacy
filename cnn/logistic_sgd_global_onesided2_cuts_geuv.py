"""
This tutorial introduces logistic regression using Theano and stochastic
gradient descent.

Logistic regression is a probabilistic, linear classifier. It is parametrized
by a weight matrix :math:`W` and a bias vector :math:`b`. Classification is
done by projecting data points onto a set of hyperplanes, the distance to
which is used to determine a class membership probability.

Mathematically, this can be written as:

.. math::
  P(Y=i|x, W,b) &= softmax_i(W x + b) \\
				&= \frac {e^{W_i x + b_i}} {\sum_j e^{W_j x + b_j}}


The output of the model or prediction is then done by taking the argmax of
the vector whose i'th element is P(Y=i|x).

.. math::

  y_{pred} = argmax_i P(Y=i|x,W,b)


This tutorial presents a stochastic gradient descent optimization method
suitable for large datasets.


References:

	- textbooks: "Pattern Recognition and Machine Learning" -
				 Christopher M. Bishop, section 4.3.2

"""
__docformat__ = 'restructedtext en'

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
		self.store_w(self.store_as_w_file, W)
		self.store_b(self.store_as_b_file, b)
	
	def load_w(self, w_file):
		return numpy.load(w_file)
	
	def load_b(self, b_file):
		return numpy.load(b_file)

	def __init__(self, input, L_input, n_in, n_out, load_model = False, w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
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
		
		self.w_file = w_file
		self.b_file = b_file
		self.store_as_w_file = w_file
		self.store_as_b_file = b_file
		if store_as_w_file is not None and store_as_b_file is not None :
			self.store_as_w_file = store_as_w_file
			self.store_as_b_file = store_as_b_file

		
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

		self.prox_vs_all = T.concatenate([ (T.sum(self.p_y_given_x[:, 50:90], axis=1) / T.sum(self.p_y_given_x[:, :], axis=1 )).reshape((50, 1)) , (1.0 - T.sum(self.p_y_given_x[:, 50:90], axis=1) / T.sum(self.p_y_given_x[:, :], axis=1 )).reshape((50, 1)) ], axis=1)


		'''
		self.sym_prx_up_vs_down_and_distal = T.concatenate([ (T.sum(self.p_y_given_x[:, 60:80], axis=1) / ( T.sum(self.p_y_given_x[:, 60:80], axis=1) + T.sum(self.p_y_given_x[:, 145:165], axis=1) + self.p_y_given_x[:, 185] )).reshape((50, 1)) , (1.0 - T.sum(self.p_y_given_x[:, 60:80], axis=1) / ( T.sum(self.p_y_given_x[:, 60:80], axis=1) + T.sum(self.p_y_given_x[:, 145:165], axis=1) + self.p_y_given_x[:, 185] )).reshape((50, 1)) ], axis=1)

		self.tomm5_p_y_given_x = T.concatenate([self.p_y_given_x[:, 30:80] / ( T.sum(self.p_y_given_x[:, 30:80], axis=1) + self.p_y_given_x[:, 185] ).reshape((50, 1)), (self.p_y_given_x[:, 185].reshape((50, 1)) / ( T.sum(self.p_y_given_x[:, 30:80], axis=1) + self.p_y_given_x[:, 185] ).reshape((50, 1))) ], axis=1)

		self.tomm5_up_vs_distal = T.concatenate([(T.sum(self.p_y_given_x[:, 30:80], axis=1) / ( T.sum(self.p_y_given_x[:, 30:80], axis=1) + self.p_y_given_x[:, 185] )).reshape((50, 1)), (self.p_y_given_x[:, 185] / ( T.sum(self.p_y_given_x[:, 30:80], axis=1) + self.p_y_given_x[:, 185] )).reshape((50, 1)) ], axis=1)
		'''


		self.p_train_standard = self.p_y_given_x

		#tomm5_pos_range = numpy.concatenate([numpy.arange(30, 80), numpy.arange(185, 186)]).tolist()
		#self.p_train_tomm5 = self.partial_pos(self.p_y_given_x, tomm5_pos_range, self.p_y_given_x.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		self.p_train_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		self.p_train_symprx = self.partial_versus(self.p_y_given_x, symprx_range, symprx_range_versus, self.p_y_given_x.shape[0])


		self.L_standard = T.eq( T.sum(L_input[:, [22, 30, 31, 32, 33, 34, 35]], axis=1), 1 ).nonzero()[0]
		self.L_tomm5 = T.eq( T.sum(L_input[:, [2, 5, 8, 11]], axis=1), 1 ).nonzero()[0]
		self.L_symprx = T.eq( L_input[:, 20], 1 ).nonzero()[0]


		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		self.p_valid_standard = self.partial_versus(self.p_y_given_x, standard_range, standard_range_versus, self.p_y_given_x.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#self.p_valid_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		self.p_valid_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		self.p_valid_symprx = self.partial_versus(self.p_y_given_x, symprx_range, symprx_range_versus, self.p_y_given_x.shape[0])


		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		#self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		self.params = [self.W, self.b]


	def predict_distribution(self, positions) :
		return self.partial_pos(self.p_y_given_x, positions, self.p_y_given_x.shape[0])

	def partial_pos(self, p_mat, partial_range, batch_size) :
		return p_mat[:, partial_range] / T.sum(p_mat[:, partial_range], axis=1).reshape((batch_size, 1))

	def partial_versus(self, p_mat, partial_range, partial_range_versus, batch_size) :
		partial_p = (T.sum(p_mat[:, partial_range], axis=1) / (T.sum(p_mat[:, partial_range], axis=1) + T.sum(p_mat[:, partial_range_versus], axis=1))).reshape((batch_size, 1))
		partial_versus_p = 1.0 - partial_p

		return T.concatenate([partial_p, partial_versus_p], axis=1)


	def negative_log_likelihood(self, y, L_input):
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

		#return -T.mean( T.sum(T.mul(T.log(self.p_y_given_x), y), axis=1) )



		y_train_standard = y

		#tomm5_pos_range = numpy.concatenate([numpy.arange(30, 80), numpy.arange(185, 186)]).tolist()
		#y_train_tomm5 = self.partial_pos(y, tomm5_pos_range, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_train_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_train_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])


		return - ( T.switch(self.L_standard.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_standard[self.L_standard, :]), y_train_standard[self.L_standard, :]), axis=1)), 0) \
			+ T.switch(self.L_tomm5.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_tomm5[self.L_tomm5, :]), y_train_tomm5[self.L_tomm5, :]), axis=1)), 0) \
			+ T.switch(self.L_symprx.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_symprx[self.L_symprx, :]), y_train_symprx[self.L_symprx, :]), axis=1)), 0)) / y.shape[0].astype('float64')



		'''
		sym_prx_y = T.concatenate([ (T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) , (1.0 - T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) ], axis=1)

		tomm5_y = T.concatenate([y[:, 30:80] / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] ).reshape((50, 1)), (y[:, 185].reshape((50, 1)) / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] ).reshape((50, 1))) ], axis=1)

		sym_prx = T.eq( L_input[:, 20], 1 )
		tomm5 = T.eq( T.sum(L_input[:, [2, 5, 8, 11]], axis=1), 1 )
		other = T.eq( T.sum(L_input[:, [22, 30, 31, 32, 33, 34, 35]], axis=1), 1 )


		return ( -T.sum( T.dot(T.log(self.p_y_given_x[other, :]), T.transpose(y[other, :]))[T.arange(y[other, :].shape[0]), T.arange(y[other, :].shape[0])] ) \
			+ -T.sum( T.dot(T.log(self.tomm5_p_y_given_x[tomm5, :]), T.transpose(tomm5_y[tomm5, :]))[T.arange(tomm5_y[tomm5, :].shape[0]), T.arange(tomm5_y[tomm5, :].shape[0])] ) \
			+ -T.sum( T.dot(T.log(self.sym_prx_up_vs_down_and_distal[sym_prx, :]), T.transpose(sym_prx_y[sym_prx, :]))[T.arange(sym_prx_y[sym_prx, :].shape[0]), T.arange(sym_prx_y[sym_prx, :].shape[0])] ) ) / y.shape[0].astype('float64')
		'''



		'''return -T.mean( T.dot(T.log(self.p_y_given_x[other, :]), T.transpose(y[other, :]))[T.arange(y[other, :].shape[0]), T.arange(y[other, :].shape[0])] )
			+ -T.mean( T.dot(T.log(self.tomm5_p_y_given_x[tomm5, :]), T.transpose(tomm5_y[tomm5, :]))[T.arange(tomm5_y[tomm5, :].shape[0]), T.arange(tomm5_y[tomm5, :].shape[0])] )
			+ -T.mean( T.dot(T.log(self.sym_prx_up_vs_down_and_distal[sym_prx, :]), T.transpose(sym_prx_y[sym_prx, :]))[T.arange(sym_prx_y[sym_prx, :].shape[0]), T.arange(sym_prx_y[sym_prx, :].shape[0])] )
		'''

		
		#return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		# end-snippet-2

	def rsquare(self, y, L_input):
		#return 1.0 - (T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1]) / T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1])))


		#other_y = T.concatenate([ (T.sum(y[:, 50:90], axis=1) / T.sum(y[:, :], axis=1 )).reshape((50, 1)) , (1.0 - T.sum(y[:, 50:90], axis=1) / T.sum(y[:, :], axis=1 )).reshape((50, 1)) ], axis=1)
		#return 1.0 - (T.dot(other_y[:,1] - self.prox_vs_all[:,1], other_y[:,1] - self.prox_vs_all[:,1]) / T.dot(other_y[:,1] - T.mean(other_y[:,1]), other_y[:,1] - T.mean(other_y[:,1])))


		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		y_valid_standard = self.partial_versus(y, standard_range, standard_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		return 1.0 - \
			( T.switch(self.L_standard.shape[0] > 0, T.dot(y_valid_standard[self.L_standard,0] - self.p_valid_standard[self.L_standard,0], y_valid_standard[self.L_standard,0] - self.p_valid_standard[self.L_standard,0]), 0) + T.switch(self.L_tomm5.shape[0] > 0, T.dot(y_valid_tomm5[self.L_tomm5,0] - self.p_valid_tomm5[self.L_tomm5,0], y_valid_tomm5[self.L_tomm5,0] - self.p_valid_tomm5[self.L_tomm5,0]), 0) + T.switch(self.L_symprx.shape[0] > 0, T.dot(y_valid_symprx[self.L_symprx,0] - self.p_valid_symprx[self.L_symprx,0], y_valid_symprx[self.L_symprx,0] - self.p_valid_symprx[self.L_symprx,0]), 0) ) / \
			( T.switch(self.L_standard.shape[0] > 0, T.dot(y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0]), y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0])), 0) + T.switch(self.L_tomm5.shape[0] > 0, T.dot(y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0]), y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0])), 0) + T.switch(self.L_symprx.shape[0] > 0, T.dot(y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0]), y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0])), 0) )




		'''sym_prx_y = T.concatenate([ (T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) , (1.0 - T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) ], axis=1)
		tomm5_y = T.concatenate([(T.sum(y[:, 30:80], axis=1) / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] )).reshape((50, 1)), (y[:, 185] / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] )).reshape((50, 1)) ], axis=1)

		sym_prx = T.eq( L_input[:, 20], 1 )
		tomm5 = T.eq( T.sum(L_input[:, [2, 5, 8, 11]], axis=1), 1 )
		other = T.eq( T.sum(L_input[:, [22, 30, 31, 32, 33, 34, 35]], axis=1), 1 )

		return 1.0 - \
			( T.dot(other_y[other,0] - self.prox_vs_all[other,0], other_y[other,0] - self.prox_vs_all[other,0]) + T.dot(tomm5_y[tomm5,0] - self.tomm5_up_vs_distal[tomm5,0], tomm5_y[tomm5,0] - self.tomm5_up_vs_distal[tomm5,0]) + T.dot(sym_prx_y[sym_prx,0] - self.sym_prx_up_vs_down_and_distal[sym_prx,0], sym_prx_y[sym_prx,0] - self.sym_prx_up_vs_down_and_distal[sym_prx,0]) ) / \
			( T.dot(other_y[other,0] - T.mean(other_y[other,0]), other_y[other,0] - T.mean(other_y[other,0])) + T.dot(tomm5_y[tomm5,0] - T.mean(tomm5_y[tomm5,0]), tomm5_y[tomm5,0] - T.mean(tomm5_y[tomm5,0])) + T.dot(sym_prx_y[sym_prx,0] - T.mean(sym_prx_y[sym_prx,0]), sym_prx_y[sym_prx,0] - T.mean(sym_prx_y[sym_prx,0])) )'''

	
	def sse(self, y):
		#return T.dot(y[:,1] - self.p_y_given_x[:,1], y[:,1] - self.p_y_given_x[:,1])
		
		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		y_valid_standard = self.partial_versus(y, standard_range, standard_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		return ( T.switch(self.L_standard.shape[0] > 0, T.dot(y_valid_standard[self.L_standard,0] - self.p_valid_standard[self.L_standard,0], y_valid_standard[self.L_standard,0] - self.p_valid_standard[self.L_standard,0]), 0) + T.switch(self.L_tomm5.shape[0] > 0, T.dot(y_valid_tomm5[self.L_tomm5,0] - self.p_valid_tomm5[self.L_tomm5,0], y_valid_tomm5[self.L_tomm5,0] - self.p_valid_tomm5[self.L_tomm5,0]), 0) + T.switch(self.L_symprx.shape[0] > 0, T.dot(y_valid_symprx[self.L_symprx,0] - self.p_valid_symprx[self.L_symprx,0], y_valid_symprx[self.L_symprx,0] - self.p_valid_symprx[self.L_symprx,0]), 0) )
	
	def sst(self, y):
		#return T.dot(y[:,1] - T.mean(y[:,1]), y[:,1] - T.mean(y[:,1]))
		
		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		y_valid_standard = self.partial_versus(y, standard_range, standard_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		return ( T.switch(self.L_standard.shape[0] > 0, T.dot(y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0]), y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0])), 0) + T.switch(self.L_tomm5.shape[0] > 0, T.dot(y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0]), y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0])), 0) + T.switch(self.L_symprx.shape[0] > 0, T.dot(y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0]), y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0])), 0) )
	
	def abs_error(self, y):
		return T.mean(abs(y[:,1] - self.p_y_given_x[:,1]))
	
	def recall(self):
		#return self.p_y_given_x[:,1]

		#p = T.switch(self.L_tomm5.shape[0] > 0, T.set_subtensor(self.p_valid_standard[self.L_tomm5, :], self.p_valid_tomm5[self.L_tomm5, :]), self.p_valid_standard)
		#p = T.switch(self.L_symprx.shape[0] > 0, T.set_subtensor(p[self.L_symprx, :], self.p_valid_symprx[self.L_symprx, :]), p)

		standard_range = numpy.arange(55, 85).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()

		#standard_range = numpy.arange(60, 80).tolist()
		#standard_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()

		#standard_range = numpy.arange(60, 90).tolist()
		#standard_range_versus = numpy.concatenate([numpy.arange(120, 186)]).tolist()

		p = self.partial_versus(self.p_y_given_x, standard_range, standard_range_versus, self.p_y_given_x.shape[0])

		return p[:, 0]

	def recall_y(self, y):
		#return self.p_y_given_x[:,1]

		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		y_valid_standard = self.partial_versus(y, standard_range, standard_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		p = T.switch(self.L_tomm5.shape[0] > 0, T.set_subtensor(y_valid_standard[self.L_tomm5, :], y_valid_tomm5[self.L_tomm5, :]), y_valid_standard)
		p = T.switch(self.L_symprx.shape[0] > 0, T.set_subtensor(p[self.L_symprx, :], y_valid_symprx[self.L_symprx, :]), p)

		return p[:, 0]


	def log_loss(self, y, L_input):
		#return -T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])]


		y_train_standard = y

		#tomm5_pos_range = numpy.concatenate([numpy.arange(30, 80), numpy.arange(185, 186)]).tolist()
		#y_train_tomm5 = self.partial_pos(y, tomm5_pos_range, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		y_train_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		y_train_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])


		return - ( T.switch(self.L_standard.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_standard[self.L_standard, :]), y_train_standard[self.L_standard, :]), axis=1)), 0) \
			+ T.switch(self.L_tomm5.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_tomm5[self.L_tomm5, :]), y_train_tomm5[self.L_tomm5, :]), axis=1)), 0) \
			+ T.switch(self.L_symprx.shape[0] > 0, T.sum(T.sum(T.mul(T.log(self.p_train_symprx[self.L_symprx, :]), y_train_symprx[self.L_symprx, :]), axis=1)), 0))


		'''sym_prx_y = T.concatenate([ (T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) , (1.0 - T.sum(y[:, 60:80], axis=1) / ( T.sum(y[:, 60:80], axis=1) + T.sum(y[:, 145:165], axis=1) + y[:, 185] )).reshape((50, 1)) ], axis=1)
		tomm5_y = T.concatenate([y[:, 30:80] / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] ).reshape((50, 1)), (y[:, 185].reshape((50, 1)) / ( T.sum(y[:, 30:80], axis=1) + y[:, 185] ).reshape((50, 1))) ], axis=1)

		sym_prx = T.eq( L_input[:, 20], 1 )
		tomm5 = T.eq( T.sum(L_input[:, [2, 5, 8, 11]], axis=1), 1 )
		other = T.eq( T.sum(L_input[:, [22, 30, 31, 32, 33, 34, 35]], axis=1), 1 )

		return ( -T.sum( T.dot(T.log(self.p_y_given_x[other, :]), T.transpose(y[other, :]))[T.arange(y[other, :].shape[0]), T.arange(y[other, :].shape[0])] ) \
			+ -T.sum( T.dot(T.log(self.tomm5_p_y_given_x[tomm5, :]), T.transpose(tomm5_y[tomm5, :]))[T.arange(tomm5_y[tomm5, :].shape[0]), T.arange(tomm5_y[tomm5, :].shape[0])] ) \
			+ -T.sum( T.dot(T.log(self.sym_prx_up_vs_down_and_distal[sym_prx, :]), T.transpose(sym_prx_y[sym_prx, :]))[T.arange(sym_prx_y[sym_prx, :].shape[0]), T.arange(sym_prx_y[sym_prx, :].shape[0])] ) )
		'''


def shared_dataset(data, datatype=theano.config.floatX, borrow=True):
		""" Function that loads the dataset into shared variables"""
		
		shared_data = theano.shared(numpy.asarray(data,
											   dtype=datatype),
								 borrow=borrow)

		return shared_data

def load_output_data(dataset):#,1,2
	
	y_ref = numpy.matrix(numpy.load('snps/apa_' + dataset + '_output_ref.npy'))
	y_var = numpy.matrix(numpy.load('snps/apa_' + dataset + '_output_var.npy'))

	y_ref = numpy.hstack([numpy.matrix(1.0 - numpy.sum(y_ref[:, 0],axis=1)), y_ref[:, 0]])
	y_var = numpy.hstack([numpy.matrix(1.0 - numpy.sum(y_var[:, 0],axis=1)), y_var[:, 0]])


	snptype = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snptype.npy')[:,0])
	apadist = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snppos.npy')[:,0])

	'''snptype_prune_index = (snptype == 2)
	y_ref = y_ref[snptype_prune_index,:]
	y_var = y_var[snptype_prune_index,:]
	apadist = apadist[snptype_prune_index]
	snptype = snptype[snptype_prune_index]'''

	snpregion_prune_index = (apadist >= 25) & (apadist <= 110)
	y_ref = y_ref[snpregion_prune_index,:]
	y_var = y_var[snpregion_prune_index,:]
	apadist = apadist[snpregion_prune_index]
	snptype = snptype[snpregion_prune_index]


	ref_y = theano.shared(numpy.asarray(y_ref, dtype=theano.config.floatX), borrow=True)
	var_y = theano.shared(numpy.asarray(y_var, dtype=theano.config.floatX), borrow=True)
	
	rval = [ref_y, var_y]
	return rval

def load_input_data(dataset):
	#############
	# LOAD DATA #
	#############

	print('... loading data')
	
	#X = numpy.load(dataset + '_input.npy')#_foldprob
	X = spio.loadmat('snps/apa_fullseq_' + dataset + '_input.mat')
	X_ref = sp.csr_matrix(X["ref"])
	X_var = sp.csr_matrix(X["var"])

	snptype = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snptype.npy')[:,0])
	apadist = numpy.ravel(numpy.load('snps/apa_' + dataset + '_snppos.npy')[:,0])

	'''snptype_prune_index = (snptype == 2)
	X_ref = X_ref[snptype_prune_index,:]
	X_var = X_var[snptype_prune_index,:]
	snptype = snptype[snptype_prune_index]
	apadist = apadist[snptype_prune_index]'''

	snpregion_prune_index = (apadist >= 25) & (apadist <= 110)
	X_ref = X_ref[snpregion_prune_index,:]
	X_var = X_var[snpregion_prune_index,:]
	snptype = snptype[snpregion_prune_index]
	apadist = apadist[snpregion_prune_index]


	'''X_ref = numpy.array(X_ref.todense())

	X_var = numpy.array(X_var.todense())

	pre = numpy.zeros((X_ref.shape[0], 24, 4))
	pre_str = 'XXXXXXXXXCATTACTCGCATCCA'
	for j in range(0, len(pre_str)) :
		for i in range(0, pre.shape[0]) :
			if pre_str[j] == 'A' :
				pre[i, j, 0] = 1
			elif pre_str[j] == 'C' :
				pre[i, j, 1] = 1
			elif pre_str[j] == 'G' :
				pre[i, j, 2] = 1
			elif pre_str[j] == 'T' :
				pre[i, j, 3] = 1
	pre = pre.reshape((X_ref.shape[0], 24 * 4))


	post = numpy.zeros((X_ref.shape[0], 5, 4))
	post_str = 'CTACG'
	for j in range(0, len(post_str)) :
		for i in range(0, post.shape[0]) :
			if post_str[j] == 'A' :
				post[i, j, 0] = 1
			elif post_str[j] == 'C' :
				post[i, j, 1] = 1
			elif post_str[j] == 'G' :
				post[i, j, 2] = 1
			elif post_str[j] == 'T' :
				post[i, j, 3] = 1
	post = post.reshape((X_ref.shape[0], 5 * 4))

	X_ref = numpy.concatenate([pre, X_ref[:, 25*4:X_ref.shape[1]-5*4], post], axis=1)
	X_var = numpy.concatenate([pre, X_var[:, 25*4:X_var.shape[1]-5*4], post], axis=1)

	X_ref = sp.csr_matrix(X_ref)
	X_var = sp.csr_matrix(X_var)

	print('Injected X shapes:')
	print(X_ref.shape)
	print(X_var.shape)

	for i in range(0, X_ref.shape[0]) :
		print(translate_to_seq(X_ref[i, :]))
		print(translate_to_seq(X_var[i, :]))
		print('')'''

	ref_x = theano.shared(X_ref, borrow=True)
	var_x = theano.shared(X_var, borrow=True)

	L_input = numpy.zeros((X_ref.shape[0], 36))
	zero_L = theano.shared(numpy.asarray(L_input, dtype=theano.config.floatX), borrow=True)
	
	rval = [ref_x, var_x, zero_L, apadist]
	return rval

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
	datasets = load_data()
	#sgd_optimization_mnist()
