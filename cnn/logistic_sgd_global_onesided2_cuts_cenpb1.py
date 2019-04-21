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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#self.p_train_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		self.p_train_tomm5 = self.p_y_given_x

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#self.p_train_symprx = self.partial_versus(self.p_y_given_x, symprx_range, symprx_range_versus, self.p_y_given_x.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#self.p_valid_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()
		self.p_valid_tomm5 = self.partial_versus(self.p_y_given_x, tomm5_range, tomm5_range_versus, self.p_y_given_x.shape[0])

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#self.p_valid_symprx = self.partial_versus(self.p_y_given_x, symprx_range, symprx_range_versus, self.p_y_given_x.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_train_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		y_train_tomm5 = y

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_train_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		return ( T.switch(self.L_standard.shape[0] > 0, T.dot(y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0]), y_valid_standard[self.L_standard,0] - T.mean(y_valid_standard[self.L_standard,0])), 0) + T.switch(self.L_tomm5.shape[0] > 0, T.dot(y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0]), y_valid_tomm5[self.L_tomm5,0] - T.mean(y_valid_tomm5[self.L_tomm5,0])), 0) + T.switch(self.L_symprx.shape[0] > 0, T.dot(y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0]), y_valid_symprx[self.L_symprx,0] - T.mean(y_valid_symprx[self.L_symprx,0])), 0) )
	
	def abs_error(self, y):
		return T.mean(abs(y[:,1] - self.p_y_given_x[:,1]))


	def avg_cut_from_pas(self):
		return T.dot(self.p_y_given_x[:, 60:90], T.arange(60, 90).reshape((30, 1))) / T.sum(self.p_y_given_x[:, 60:90], axis=1).reshape((self.p_y_given_x.shape[0] ,1)) - 49

	def avg_cut_from_pas_y(self, y):
		return T.dot(y[:, 60:90], T.arange(60, 90).reshape((30, 1))) / T.sum(y[:, 60:90], axis=1).reshape((y.shape[0] ,1)) - 49

	
	def recall(self):
		#return self.p_y_given_x[:,1]

		'''standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()

		#standard_range = numpy.arange(55, 85).tolist()
		#standard_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()

		return self.partial_versus(self.p_y_given_x, standard_range, standard_range_versus, self.p_y_given_x.shape[0])[:, 0]'''

		p = T.switch(self.L_tomm5.shape[0] > 0, T.set_subtensor(self.p_valid_standard[self.L_tomm5, :], self.p_valid_tomm5[self.L_tomm5, :]), self.p_valid_standard)
		p = T.switch(self.L_symprx.shape[0] > 0, T.set_subtensor(p[self.L_symprx, :], self.p_valid_symprx[self.L_symprx, :]), p)

		return p[:, 0]

	def recall_y(self, y):
		#return self.p_y_given_x[:,1]

		standard_range = numpy.arange(60, 90).tolist()
		standard_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(90, 186)]).tolist()
		y_valid_standard = self.partial_versus(y, standard_range, standard_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(50, 80).tolist()
		#tomm5_range_versus = numpy.arange(185, 186).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		tomm5_range = numpy.arange(55, 85).tolist()
		tomm5_range_versus = numpy.concatenate([numpy.arange(0, 55), numpy.arange(85, 186)]).tolist()
		y_valid_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
		y_valid_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		p = T.switch(self.L_tomm5.shape[0] > 0, T.set_subtensor(y_valid_standard[self.L_tomm5, :], y_valid_tomm5[self.L_tomm5, :]), y_valid_standard)
		p = T.switch(self.L_symprx.shape[0] > 0, T.set_subtensor(p[self.L_symprx, :], y_valid_symprx[self.L_symprx, :]), p)

		return p[:, 0]


	def log_loss(self, y, L_input):
		#return -T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])]


		y_train_standard = y

		#tomm5_pos_range = numpy.concatenate([numpy.arange(30, 80), numpy.arange(185, 186)]).tolist()
		#y_train_tomm5 = self.partial_pos(y, tomm5_pos_range, y.shape[0])

		#tomm5_range = numpy.arange(55, 85).tolist()
		#tomm5_range_versus = numpy.concatenate([numpy.arange(30, 55), numpy.arange(185, 186)]).tolist()
		#y_train_tomm5 = self.partial_versus(y, tomm5_range, tomm5_range_versus, y.shape[0])

		y_train_tomm5 = y

		#symprx_range = numpy.arange(60, 80).tolist()
		#symprx_range_versus = numpy.concatenate([numpy.arange(145, 165), numpy.arange(185, 186)]).tolist()
		#y_train_symprx = self.partial_versus(y, symprx_range, symprx_range_versus, y.shape[0])

		symprx_range = numpy.arange(60, 80).tolist()
		symprx_range_versus = numpy.concatenate([numpy.arange(0, 60), numpy.arange(80, 186)]).tolist()
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

def load_output_data(dataset, shuffle_index=None, count_filter=None, balance_all_libraries=False):
	#############
	# LOAD DATA #
	#############

	print('... loading data')

	write_cache = False
	use_cache = False

	cache_name = '22_5000'#'22_5000'#'20'#'20_AATAAA_pPas_No_dPas'#'20'#'22_denovo'#'22'#'20_AATAAA_pPas_No_dPas'#'20_AATAAA'#'22'#'20'#'test2percent'#''#20'#'20_22'

	#train_set_split = 0.95
	#valid_set_split = 0.02
	train_set_split = 0.96
	valid_set_split = 0.02

	constant_set_split = True
	constant_valid_set_split = 100
	constant_test_set_split = 35000#5000

	rval = None

	if use_cache == False :
	
		Y = spio.loadmat('apa_' + dataset + '_cutdistribution.mat')
		Y = sp.csr_matrix(Y["cuts"])

		c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))
		L = numpy.ravel(numpy.load('apa_' + dataset + '_libindex.npy'))

		#Only non-competing PASes
		'''Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
		F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
		F_aataaa = numpy.ravel(F[:, 0].todense())
		F_attaaa = numpy.ravel(F[:, 1].todense())
		F_denovo = numpy.ravel(F[:, 5].todense())

		F_dpas_aataaa = numpy.ravel(F[:, 2].todense())
		F_dpas_attaaa = numpy.ravel(F[:, 3].todense())

		Y = Y[(F_aataaa == 1) & (F_dpas_aataaa == 0),:]
		L = L[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		c = c[(F_aataaa == 1) & (F_dpas_aataaa == 0)]'''

		if count_filter is not None and count_filter > 0 :
			c_keep_index = numpy.nonzero(c > count_filter)[0]

			Y = Y[c_keep_index,:]
			L = L[c_keep_index]
			c = c[c_keep_index]

		print('Done count filtering.')
		print(Y.shape)


		'''print('Y_debug')
		Y_debug = Y[L <= 11, :]
		Y_debug = numpy.ravel(Y_debug[:, 0:140].sum(axis=1)) + numpy.ravel(Y_debug[:, 185].todense())
		Y_debug_sort_index = numpy.argsort(Y_debug)
		print(Y_debug[Y_debug_sort_index])
		print('' + 1)'''


		if balance_all_libraries == True :

			#List of included libraries
			L_included = [22, 30, 31, 32, 33, 34, 35]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 11, 20, 22, 31, 32, 33, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 11, 20, 22, 31]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]

			arranged_index_len = 0
			min_join_len = len(numpy.nonzero(L == L_included[0])[0])

			for lib in L_included :
				lib_len = len(numpy.nonzero(L == lib)[0])
				arranged_index_len += lib_len
				if lib_len < min_join_len :
					min_join_len = lib_len

			arranged_index = numpy.zeros(arranged_index_len, dtype=numpy.int)

			arranged_remainder_index = 0
			arranged_join_index = arranged_index_len - len(L_included) * min_join_len

			for lib_i in range(0, len(L_included)) :
				lib = L_included[lib_i]

				print('Arranging lib ' + str(lib))

				#1. Get indexes of each Library

				apa_lib_index = numpy.nonzero(L == lib)[0]

				#2. Sort indexes of each library by count
				c_apa_lib = c[apa_lib_index]
				sort_index_apa_lib = numpy.argsort(c_apa_lib)
				apa_lib_index = apa_lib_index[sort_index_apa_lib]

				#3. Shuffle indexes of each library modulo 2
				even_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 0
				odd_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 1

				apa_lib_index_even = apa_lib_index[even_index_apa_lib]
				apa_lib_index_odd = apa_lib_index[odd_index_apa_lib]

				apa_lib_index = numpy.concatenate([apa_lib_index_even, apa_lib_index_odd])

				#4. Join modulo 2
				i = 0
				for j in range(len(apa_lib_index) - min_join_len, len(apa_lib_index)) :
					arranged_index[arranged_join_index + i * len(L_included) + lib_i] = apa_lib_index[j]
					i += 1

				#5. Append remainder
				for j in range(0, len(apa_lib_index) - min_join_len) :
					arranged_index[arranged_remainder_index] = apa_lib_index[j]
					arranged_remainder_index += 1

			if shuffle_index is not None :
				if constant_set_split == False :
					arranged_index[:int(train_set_split * len(arranged_index))] = arranged_index[shuffle_index]
				else :
					arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)] = arranged_index[shuffle_index]

			print('Arranged index:')
			print(len(arranged_index))
			print(arranged_index)

			Y = Y[arranged_index,:]
			L = L[arranged_index]
			c = c[arranged_index]


		Y_train = None
		Y_validation = None
		Y_test = None
		if constant_set_split == False :
			Y_train = Y[:int(train_set_split * Y.shape[0]),:]
			Y_validation = Y[Y_train.shape[0]:Y_train.shape[0] + int(valid_set_split * Y.shape[0]),:]
			Y_test = Y[Y_train.shape[0] + Y_validation.shape[0]:,:]
		else :
			Y_train = Y[:Y.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			Y_validation = Y[Y.shape[0] - constant_valid_set_split - constant_test_set_split:Y.shape[0] - constant_test_set_split,:]
			Y_test = Y[Y.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		Y_train = sp.csr_matrix(numpy.zeros((100, Y.shape[1])))

		train_set_y = theano.shared(Y_train, borrow=True)
		valid_set_y = theano.shared(Y_validation, borrow=True)
		test_set_y = theano.shared(Y_test, borrow=True)

		if write_cache == True :
			numpy.savez('cache_' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_train', data=Y_train.data, indices=Y_train.indices, indptr=Y_train.indptr, shape=Y_train.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_validation', data=Y_validation.data, indices=Y_validation.indices, indptr=Y_validation.indptr, shape=Y_validation.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_test', data=Y_test.data, indices=Y_test.indices, indptr=Y_test.indptr, shape=Y_test.shape)

		
		rval = [train_set_y, valid_set_y, test_set_y]
	
	else :

		if cache_name != '' :
			cache_name = '_' + cache_name

		loader = numpy.load('cache' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_train.npz')
		Y_train = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_validation.npz')
		Y_validation = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_' + dataset + '_cutdistribution_test.npz')
		Y_test = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)

		train_set_y = theano.shared(Y_train, borrow=True)
		valid_set_y = theano.shared(Y_validation, borrow=True)
		test_set_y = theano.shared(Y_test, borrow=True)

		rval = [train_set_y, valid_set_y, test_set_y]

	return rval



def load_input_data(dataset, shuffle=False, count_filter=None, balance_all_libraries=False):
	#############
	# LOAD DATA #
	#############

	print('... loading data')

	write_cache = False
	use_cache = False

	cache_name = '22_5000'#'22_5000'#'20'#'20_AATAAA_pPas_No_dPas'#'20'#'22_denovo'#'22'#'20_AATAAA_pPas_No_dPas'#'20_AATAAA'#'22'#'20'#'test2percent'#''#20'#'20_22'

	#train_set_split = 0.95
	#valid_set_split = 0.02
	train_set_split = 0.96
	valid_set_split = 0.02

	constant_set_split = True
	constant_valid_set_split = 100
	constant_test_set_split = 35000#5000

	rval = None

	shuffle_index = None

	if use_cache == False :
	
		#X = numpy.load(dataset + '_input.npy')#_foldprob
		#X = spio.loadmat('apa_fullseq_v_' + dataset + '_input.mat')
		#X = sp.csr_matrix(X["input"])

		loader = numpy.load('npz_apa_seq_' + dataset + '_input.npz')
		#loader = numpy.load('npz_apa_fullseq_' + dataset + '_input.npz')
		X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)


		c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))
		L = numpy.ravel(numpy.load('apa_' + dataset + '_libindex.npy'))

		d_input = numpy.load('apa_' + dataset + '_distalpas.npy')

		#Only non-competing PASes
		'''Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
		F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
		F_aataaa = numpy.ravel(F[:, 0].todense())
		F_attaaa = numpy.ravel(F[:, 1].todense())
		F_denovo = numpy.ravel(F[:, 5].todense())

		F_dpas_aataaa = numpy.ravel(F[:, 2].todense())
		F_dpas_attaaa = numpy.ravel(F[:, 3].todense())

		X = X[(F_aataaa == 1) & (F_dpas_aataaa == 0),:]
		L = L[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		c = c[(F_aataaa == 1) & (F_dpas_aataaa == 0)]
		d_input = d_input[(F_aataaa == 1),:]'''

		if count_filter is not None and count_filter > 0 :
			c_keep_index = numpy.nonzero(c > count_filter)[0]

			X = X[c_keep_index,:]
			L = L[c_keep_index]
			d_input = d_input[c_keep_index,:]
			c = c[c_keep_index]

		print('Done count filtering.')
		print(X.shape)

		if balance_all_libraries == True :

			#List of included libraries
			L_included = [22, 30, 31, 32, 33, 34, 35]#[22]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 11, 20, 22, 31, 32, 33, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[2, 5, 11, 20, 22, 31]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]

			arranged_index_len = 0
			min_join_len = len(numpy.nonzero(L == L_included[0])[0])

			for lib in L_included :
				lib_len = len(numpy.nonzero(L == lib)[0])
				arranged_index_len += lib_len
				if lib_len < min_join_len :
					min_join_len = lib_len

			arranged_index = numpy.zeros(arranged_index_len, dtype=numpy.int)

			arranged_remainder_index = 0
			arranged_join_index = arranged_index_len - len(L_included) * min_join_len

			for lib_i in range(0, len(L_included)) :
				lib = L_included[lib_i]

				print('Arranging lib ' + str(lib))

				#1. Get indexes of each Library

				apa_lib_index = numpy.nonzero(L == lib)[0]

				#2. Sort indexes of each library by count
				c_apa_lib = c[apa_lib_index]
				sort_index_apa_lib = numpy.argsort(c_apa_lib)
				apa_lib_index = apa_lib_index[sort_index_apa_lib]

				#3. Shuffle indexes of each library modulo 2
				even_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 0
				odd_index_apa_lib = numpy.arange(len(apa_lib_index)) % 2 == 1

				apa_lib_index_even = apa_lib_index[even_index_apa_lib]
				apa_lib_index_odd = apa_lib_index[odd_index_apa_lib]

				apa_lib_index = numpy.concatenate([apa_lib_index_even, apa_lib_index_odd])

				#4. Join modulo 2
				i = 0
				for j in range(len(apa_lib_index) - min_join_len, len(apa_lib_index)) :
					arranged_index[arranged_join_index + i * len(L_included) + lib_i] = apa_lib_index[j]
					i += 1

				#5. Append remainder
				for j in range(0, len(apa_lib_index) - min_join_len) :
					arranged_index[arranged_remainder_index] = apa_lib_index[j]
					arranged_remainder_index += 1

			if shuffle == True :
				if constant_set_split == False :
					shuffle_index = numpy.arange(len(arranged_index[:int(train_set_split * len(arranged_index))]))
					numpy.random.shuffle(shuffle_index)
					arranged_index[:int(train_set_split * len(arranged_index))] = arranged_index[shuffle_index]
				else :
					shuffle_index = numpy.arange(len(arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)]))
					numpy.random.shuffle(shuffle_index)
					arranged_index[:int(len(arranged_index) - constant_valid_set_split - constant_test_set_split)] = arranged_index[shuffle_index]

			print('Arranged index:')
			print(len(arranged_index))
			print(arranged_index)

			X = X[arranged_index,:]
			L = L[arranged_index]
			d_input = d_input[arranged_index,:]
			c = c[arranged_index]


		L_input = numpy.zeros((len(c), 36))
		for i in range(0, len(c)) :
			L_input[i,int(L[i])] = 1

		c_input = numpy.zeros((len(c), 1))
		for i in range(0, len(c)) :
			c_input[i, 0] = c[i]

		X_train = None
		X_validation = None
		X_test = None
		if constant_set_split == False :
			X_train = X[:int(train_set_split * X.shape[0]),:]
			X_validation = X[X_train.shape[0]:X_train.shape[0] + int(valid_set_split * X.shape[0]),:]
			X_test = X[X_train.shape[0] + X_validation.shape[0]:,:]
		else :
			X_train = X[:X.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			X_validation = X[X.shape[0] - constant_valid_set_split - constant_test_set_split:X.shape[0] - constant_test_set_split,:]
			X_test = X[X.shape[0] - constant_test_set_split:,:]
		#Subselect of test set
		#X_test = X_test[y_test >= 0.8,:]

		#Spoof training set
		X_train = sp.csr_matrix(numpy.zeros((100, X.shape[1])))

		train_set_x = theano.shared(X_train, borrow=True)
		valid_set_x = theano.shared(X_validation, borrow=True)
		test_set_x = theano.shared(X_test, borrow=True)

		L_train = None
		L_validation = None
		L_test = None
		if constant_set_split == False :
			L_train = L_input[:int(train_set_split * L_input.shape[0]),:]
			L_validation = L_input[L_train.shape[0]:L_train.shape[0] + int(valid_set_split * L_input.shape[0]),:]
			L_test = L_input[L_train.shape[0] + L_validation.shape[0]:,:]
		else :
			L_train = L_input[:L_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			L_validation = L_input[L_input.shape[0] - constant_valid_set_split - constant_test_set_split:L_input.shape[0] - constant_test_set_split,:]
			L_test = L_input[L_input.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		L_train = numpy.zeros((100, 30))

		train_set_L = theano.shared(numpy.asarray(L_train, dtype=theano.config.floatX), borrow=True)
		valid_set_L = theano.shared(numpy.asarray(L_validation, dtype=theano.config.floatX), borrow=True)
		test_set_L = theano.shared(numpy.asarray(L_test, dtype=theano.config.floatX), borrow=True)

		d_train = None
		d_validation = None
		d_test = None
		if constant_set_split == False :
			d_train = d_input[:int(train_set_split * d_input.shape[0]),:]
			d_validation = d_input[d_train.shape[0]:d_train.shape[0] + int(valid_set_split * d_input.shape[0]),:]
			d_test = d_input[d_train.shape[0] + d_validation.shape[0]:,:]
		else :
			d_train = d_input[:d_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			d_validation = d_input[d_input.shape[0] - constant_valid_set_split - constant_test_set_split:d_input.shape[0] - constant_test_set_split,:]
			d_test = d_input[d_input.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		d_train = numpy.zeros((100, 30))

		train_set_d = theano.shared(numpy.asarray(d_train, dtype=theano.config.floatX), borrow=True)
		valid_set_d = theano.shared(numpy.asarray(d_validation, dtype=theano.config.floatX), borrow=True)
		test_set_d = theano.shared(numpy.asarray(d_test, dtype=theano.config.floatX), borrow=True)

		c_train = None
		c_validation = None
		c_test = None
		if constant_set_split == False :
			c_train = c_input[:int(train_set_split * c_input.shape[0]),:]
			c_validation = c_input[c_train.shape[0]:c_train.shape[0] + int(valid_set_split * c_input.shape[0]),:]
			c_test = c_input[c_train.shape[0] + c_validation.shape[0]:,:]
		else :
			c_train = c_input[:c_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			c_validation = c_input[c_input.shape[0] - constant_valid_set_split - constant_test_set_split:c_input.shape[0] - constant_test_set_split,:]
			c_test = c_input[c_input.shape[0] - constant_test_set_split:,:]

		#Spoof training set
		c_train = numpy.zeros((100, 1))

		train_set_c = theano.shared(numpy.asarray(c_train, dtype=theano.config.floatX), borrow=True)
		valid_set_c = theano.shared(numpy.asarray(c_validation, dtype=theano.config.floatX), borrow=True)
		test_set_c = theano.shared(numpy.asarray(c_test, dtype=theano.config.floatX), borrow=True)

		if write_cache == True :
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_train', data=X_train.data, indices=X_train.indices, indptr=X_train.indptr, shape=X_train.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_validation', data=X_validation.data, indices=X_validation.indices, indptr=X_validation.indptr, shape=X_validation.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_test', data=X_test.data, indices=X_test.indices, indptr=X_test.indptr, shape=X_test.shape)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_train', L_train)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_validation', L_validation)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_test', L_test)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_d_input_train', d_train)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_d_input_validation', d_validation)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_d_input_test', d_test)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_shuffle_index', shuffle_index)

		
		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None, train_set_d, valid_set_d, test_set_d, train_set_c, valid_set_c, test_set_c]
	
	else :

		if cache_name != '' :
			cache_name = '_' + cache_name

		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_train.npz')
		X_train = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_validation.npz')
		X_validation = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)
		loader = numpy.load('cache' + cache_name + '_npz_apa_seq_' + dataset + '_input_test.npz')
		X_test = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'], dtype=numpy.int8)

		train_set_x = theano.shared(X_train, borrow=True)
		valid_set_x = theano.shared(X_validation, borrow=True)
		test_set_x = theano.shared(X_test, borrow=True)

		L_train = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_train.npy')
		L_validation = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_validation.npy')
		L_test = numpy.load('cache' + cache_name + '_apa_' + dataset + '_L_input_test.npy')

		train_set_L = theano.shared(numpy.asarray(L_train, dtype=theano.config.floatX), borrow=True)
		valid_set_L = theano.shared(numpy.asarray(L_validation, dtype=theano.config.floatX), borrow=True)
		test_set_L = theano.shared(numpy.asarray(L_test, dtype=theano.config.floatX), borrow=True)

		d_train = numpy.load('cache' + cache_name + '_apa_' + dataset + '_d_input_train.npy')
		d_validation = numpy.load('cache' + cache_name + '_apa_' + dataset + '_d_input_validation.npy')
		d_test = numpy.load('cache' + cache_name + '_apa_' + dataset + '_d_input_test.npy')

		train_set_d = theano.shared(numpy.asarray(d_train, dtype=theano.config.floatX), borrow=True)
		valid_set_d = theano.shared(numpy.asarray(d_validation, dtype=theano.config.floatX), borrow=True)
		test_set_d = theano.shared(numpy.asarray(d_test, dtype=theano.config.floatX), borrow=True)

		shuffle_index = numpy.ravel(numpy.load('cache' + cache_name + '_apa_' + dataset + '_shuffle_index.npy'))

		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None, train_set_d, valid_set_d, test_set_d]

	return rval

if __name__ == '__main__':
	datasets = load_data()
	#sgd_optimization_mnist()
