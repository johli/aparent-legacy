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

	def __init__(self, input, n_in, n_out, load_model = False, w_file = '', b_file = '', store_as_w_file = None, store_as_b_file = None):
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

		# symbolic description of how to compute prediction as class whose
		# probability is maximal
		#self.y_pred = T.argmax(self.p_y_given_x, axis=1)
		# end-snippet-1

		# parameters of the model
		self.params = [self.W, self.b]

	def negative_log_likelihood(self, y):
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
		
		return -T.mean(T.dot(T.log(self.p_y_given_x), T.transpose(y))[T.arange(y.shape[0]), T.arange(y.shape[0])])
		
		#return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
		# end-snippet-2

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

def load_output_data(dataset, splice_site_indexes=[1], shuffle_index=None, balance_data=False, balance_test_set=False, count_filter=None, misprime_index=None, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=False):#,1,2
	
	train_set_split = 0.01 #0.96 #0.95
	valid_set_split = 0.01 #0.02 #0.02

	#constant_set_split = False
	#constant_valid_set_split = 1000
	#constant_test_set_split = 20000

	y = numpy.matrix(numpy.load('apa_' + dataset + '_output.npy'))#NOT .T
	
	y = numpy.hstack([numpy.matrix(1.0 - numpy.sum(y[:, splice_site_indexes],axis=1)), y[:, splice_site_indexes]])
	
	print(y.shape)
	print(y)
	
	c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))

	L = numpy.zeros(len(c))

	#L_prune_index = (L != 13) & (L != 14)
	#L_prune_index = L == 5
	#L_prune_index = L >= 20
	#L_prune_index = L <= 5
	#L_prune_index = (L >= 5) & (L <= 12)
	'''y = y[L_prune_index,:]
	c = c[L_prune_index]
	L = L[L_prune_index]'''

	'''sort_index = numpy.argsort(c)
	y = y[sort_index,:]
	c = c[sort_index]
	L = L[sort_index]'''

	'''if shuffle_index is not None :
		y = y[shuffle_index,:]
		c = c[shuffle_index]'''

	#Only non-competing PASes
	'''Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
	F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
	F_aataaa = numpy.ravel(F[:, 0].todense())
	F_attaaa = numpy.ravel(F[:, 1].todense())
	F_denovo = numpy.ravel(F[:, 5].todense())
	y = y[(F_denovo == 1),:]
	L = L[(F_denovo == 1)]
	c = c[(F_denovo == 1)]'''


	if count_filter is not None :
		y = y[c >= count_filter,:]
		L = L[c >= count_filter]
		c = c[c >= count_filter]

	constant_set_split = True
	constant_valid_set_split = 1
	constant_test_set_split = y.shape[0] - 2

	if balance_all_libraries == True :

		#List of included libraries
		L_included = [20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]

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

		L = L[arranged_index]
		c = c[arranged_index]
		y = y[arranged_index, :]

	if balance_libraries == True :
		#1. Get indexes of each library

		apa_fr_index = numpy.nonzero(L < 20)[0]
		apa_sym_prx_index = numpy.nonzero(L >= 20)[0]

		#2. Sort indexes of each library by count

		c_apa_fr = c[apa_fr_index]
		c_apa_sym_prx = c[apa_sym_prx_index]

		sort_index_apa_fr = numpy.argsort(c_apa_fr)
		sort_index_apa_sym_prx = numpy.argsort(c_apa_sym_prx)

		apa_fr_index = apa_fr_index[sort_index_apa_fr]
		apa_sym_prx_index = apa_sym_prx_index[sort_index_apa_sym_prx]

		#3. Shuffle indexes of each library modulo 2

		even_index_apa_fr = numpy.arange(len(apa_fr_index)) % 2 == 0
		odd_index_apa_fr = numpy.arange(len(apa_fr_index)) % 2 == 1

		even_index_apa_sym_prx = numpy.arange(len(apa_sym_prx_index)) % 2 == 0
		odd_index_apa_sym_prx = numpy.arange(len(apa_sym_prx_index)) % 2 == 1

		apa_fr_index_even = apa_fr_index[even_index_apa_fr]
		apa_fr_index_odd = apa_fr_index[odd_index_apa_fr]

		apa_sym_prx_index_even = apa_sym_prx_index[even_index_apa_sym_prx]
		apa_sym_prx_index_odd = apa_sym_prx_index[odd_index_apa_sym_prx]

		apa_fr_index = numpy.concatenate([apa_fr_index_even, apa_fr_index_odd])
		apa_sym_prx_index = numpy.concatenate([apa_sym_prx_index_even, apa_sym_prx_index_odd])

		#4. Join modulo 2

		#arranged_index = []
		#for i in range()


		arranged_index = numpy.zeros(len(apa_fr_index) + len(apa_sym_prx_index), dtype=numpy.int)
		gi = 0
		for i in range(0, len(apa_sym_prx_index) - len(apa_fr_index)) :
			arranged_index[gi] = int(apa_sym_prx_index[i])
			gi += 1
		for i in range(0, len(apa_fr_index)) :
			arranged_index[gi + i * 2] = int(apa_fr_index[i])
		li = 0
		for i in range(len(apa_sym_prx_index) - len(apa_fr_index), len(apa_sym_prx_index)) :
			arranged_index[gi + li * 2 + 1] = int(apa_sym_prx_index[i])
			li += 1

		print('Arranged index:')
		print(len(arranged_index))
		print(arranged_index)

		L = L[arranged_index]
		c = c[arranged_index]
		y = y[arranged_index, :]

	if balance_sublibraries == True :
		#1. Get indexes of each library

		apa_fr_2_index = numpy.nonzero(L == 2)[0]
		apa_fr_5_index = numpy.nonzero(L == 5)[0]
		apa_fr_8_index = numpy.nonzero(L == 8)[0]
		apa_fr_11_index = numpy.nonzero(L == 11)[0]
		apa_fr_20_index = numpy.nonzero(L == 20)[0]
		apa_fr_21_index = numpy.nonzero(L == 21)[0]

		#2. Sort indexes of each library by count

		c_apa_fr_2 = c[apa_fr_2_index]
		c_apa_fr_5 = c[apa_fr_5_index]
		c_apa_fr_8 = c[apa_fr_8_index]
		c_apa_fr_11 = c[apa_fr_11_index]
		c_apa_fr_20 = c[apa_fr_20_index]
		c_apa_fr_21 = c[apa_fr_21_index]

		sort_index_apa_fr_2 = numpy.argsort(c_apa_fr_2)
		sort_index_apa_fr_5 = numpy.argsort(c_apa_fr_5)
		sort_index_apa_fr_8 = numpy.argsort(c_apa_fr_8)
		sort_index_apa_fr_11 = numpy.argsort(c_apa_fr_11)
		sort_index_apa_fr_20 = numpy.argsort(c_apa_fr_20)
		sort_index_apa_fr_21 = numpy.argsort(c_apa_fr_21)

		apa_fr_2_index = apa_fr_2_index[sort_index_apa_fr_2]
		apa_fr_5_index = apa_fr_5_index[sort_index_apa_fr_5]
		apa_fr_8_index = apa_fr_8_index[sort_index_apa_fr_8]
		apa_fr_11_index = apa_fr_11_index[sort_index_apa_fr_11]
		apa_fr_20_index = apa_fr_20_index[sort_index_apa_fr_20]
		apa_fr_21_index = apa_fr_21_index[sort_index_apa_fr_21]

		#3. Shuffle indexes of each library modulo 2

		even_index_apa_fr_2 = numpy.arange(len(apa_fr_2_index)) % 2 == 0
		odd_index_apa_fr_2 = numpy.arange(len(apa_fr_2_index)) % 2 == 1
		even_index_apa_fr_5 = numpy.arange(len(apa_fr_5_index)) % 2 == 0
		odd_index_apa_fr_5 = numpy.arange(len(apa_fr_5_index)) % 2 == 1
		even_index_apa_fr_8 = numpy.arange(len(apa_fr_8_index)) % 2 == 0
		odd_index_apa_fr_8 = numpy.arange(len(apa_fr_8_index)) % 2 == 1
		even_index_apa_fr_11 = numpy.arange(len(apa_fr_11_index)) % 2 == 0
		odd_index_apa_fr_11 = numpy.arange(len(apa_fr_11_index)) % 2 == 1

		even_index_apa_fr_20 = numpy.arange(len(apa_fr_20_index)) % 2 == 0
		odd_index_apa_fr_20 = numpy.arange(len(apa_fr_20_index)) % 2 == 1
		even_index_apa_fr_21 = numpy.arange(len(apa_fr_21_index)) % 2 == 0
		odd_index_apa_fr_21 = numpy.arange(len(apa_fr_21_index)) % 2 == 1

		apa_fr_2_index_even = apa_fr_2_index[even_index_apa_fr_2]
		apa_fr_2_index_odd = apa_fr_2_index[odd_index_apa_fr_2]
		apa_fr_5_index_even = apa_fr_5_index[even_index_apa_fr_5]
		apa_fr_5_index_odd = apa_fr_5_index[odd_index_apa_fr_5]
		apa_fr_8_index_even = apa_fr_8_index[even_index_apa_fr_8]
		apa_fr_8_index_odd = apa_fr_8_index[odd_index_apa_fr_8]
		apa_fr_11_index_even = apa_fr_11_index[even_index_apa_fr_11]
		apa_fr_11_index_odd = apa_fr_11_index[odd_index_apa_fr_11]

		apa_fr_20_index_even = apa_fr_20_index[even_index_apa_fr_20]
		apa_fr_20_index_odd = apa_fr_20_index[odd_index_apa_fr_20]
		apa_fr_21_index_even = apa_fr_21_index[even_index_apa_fr_21]
		apa_fr_21_index_odd = apa_fr_21_index[odd_index_apa_fr_21]

		apa_fr_2_index = numpy.concatenate([apa_fr_2_index_even, apa_fr_2_index_odd])
		apa_fr_5_index = numpy.concatenate([apa_fr_5_index_even, apa_fr_5_index_odd])
		apa_fr_8_index = numpy.concatenate([apa_fr_8_index_even, apa_fr_8_index_odd])
		apa_fr_11_index = numpy.concatenate([apa_fr_11_index_even, apa_fr_11_index_odd])
		apa_fr_20_index = numpy.concatenate([apa_fr_20_index_even, apa_fr_20_index_odd])
		apa_fr_21_index = numpy.concatenate([apa_fr_21_index_even, apa_fr_21_index_odd])

		#4. Join modulo 2

		#arranged_index = []
		#for i in range()


		arranged_index = numpy.zeros(len(apa_fr_2_index) + len(apa_fr_5_index) + len(apa_fr_8_index) + len(apa_fr_11_index) + len(apa_fr_20_index) + len(apa_fr_21_index), dtype=numpy.int)

		min_shuffle = min(min(min(min(min(len(apa_fr_2_index), len(apa_fr_5_index)), len(apa_fr_8_index)), len(apa_fr_11_index)), len(apa_fr_20_index)), len(apa_fr_21_index))

		gi = 0
		for i in range(0, len(apa_fr_2_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_2_index[i])
			gi += 1
		for i in range(0, len(apa_fr_5_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_5_index[i])
			gi += 1
		for i in range(0, len(apa_fr_8_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_8_index[i])
			gi += 1
		for i in range(0, len(apa_fr_11_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_11_index[i])
			gi += 1
		for i in range(0, len(apa_fr_20_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_20_index[i])
			gi += 1
		for i in range(0, len(apa_fr_21_index) - min_shuffle) :
			arranged_index[gi] = int(apa_fr_21_index[i])
			gi += 1

		li = 0
		for i in range(len(apa_fr_2_index) - min_shuffle, len(apa_fr_2_index)) :
			arranged_index[gi + li * 6] = int(apa_fr_2_index[i])
			li += 1
		li = 0
		for i in range(len(apa_fr_5_index) - min_shuffle, len(apa_fr_5_index)) :
			arranged_index[gi + li * 6 + 1] = int(apa_fr_5_index[i])
			li += 1
		li = 0
		for i in range(len(apa_fr_8_index) - min_shuffle, len(apa_fr_8_index)) :
			arranged_index[gi + li * 6 + 2] = int(apa_fr_8_index[i])
			li += 1
		li = 0
		for i in range(len(apa_fr_11_index) - min_shuffle, len(apa_fr_11_index)) :
			arranged_index[gi + li * 6 + 3] = int(apa_fr_11_index[i])
			li += 1
		li = 0
		for i in range(len(apa_fr_20_index) - min_shuffle, len(apa_fr_20_index)) :
			arranged_index[gi + li * 6 + 4] = int(apa_fr_20_index[i])
			li += 1
		li = 0
		for i in range(len(apa_fr_21_index) - min_shuffle, len(apa_fr_21_index)) :
			arranged_index[gi + li * 6 + 5] = int(apa_fr_21_index[i])
			li += 1

		print('Arranged index:')
		print(len(arranged_index))
		print(arranged_index)

		L = L[arranged_index]
		c = c[arranged_index]
		y = y[arranged_index, :]

	if balance_data == True :
		even_index = numpy.arange(y.shape[0]) % 2 == 0
		y_even = y[even_index,:]
		c_even = c[even_index]

		odd_index = numpy.arange(y.shape[0]) % 2 == 1
		y_odd = y[odd_index,:]
		c_odd = c[odd_index]

		y = numpy.concatenate([y_even, y_odd],axis=0)
		c = numpy.concatenate([c_even, c_odd])

	y_train = None
	y_validation = None
	y_test = None
	if constant_set_split == False :
		y_train = y[:int(train_set_split * y.shape[0]),:]
		y_validation = y[y_train.shape[0]:y_train.shape[0] + int(valid_set_split * y.shape[0]),:]
		y_test = y[y_train.shape[0] + y_validation.shape[0]:,:]
	else :
		y_train = y[:y.shape[0] - constant_valid_set_split - constant_test_set_split,:]
		y_validation = y[y.shape[0] - constant_valid_set_split - constant_test_set_split:y.shape[0] - constant_test_set_split,:]
		y_test = y[y.shape[0] - constant_test_set_split:,:]

	print(y_train.shape[0])
	print(y_validation.shape[0])
	print(y_test.shape[0])

	y_test_hat = numpy.asarray(y_test[:, 1])
	print('Test set distrib [<= 0.2, 0.2 - 0.8, >= 0.8].')
	print( len(y_test_hat[y_test_hat <= 0.2]) )
	print( len(y_test_hat[(y_test_hat > 0.2) & (y_test_hat < 0.8)]) )
	print( len(y_test_hat[y_test_hat >= 0.8]) )

	print('Test set distrib [<= 0.5, > 0.5].')
	print( len(y_test_hat[y_test_hat <= 0.5]) )
	print( len(y_test_hat[y_test_hat > 0.5]) )

	#Spoof training set
	y_train = y[0:100, :]

	train_set_y = shared_dataset(y_train)
	valid_set_y = shared_dataset(y_validation)
	test_set_y = shared_dataset(y_test)

	rval = [train_set_y, valid_set_y, test_set_y]
	return rval

def permute_sparse_matrix(M, order):
	permuted_row = order[M.row]
	permuted_col = order[M.col]
	new_M = sparse.coo_matrix((M.data, (permuted_row, permuted_col)), shape=M.shape)
	return new_M

def load_input_data(dataset, prefix='prox', shuffle=False, balance_data=False, balance_test_set=False, count_filter=None, misprime_filter=False, balance_libraries=False, balance_sublibraries=False, balance_all_libraries=False):
	#############
	# LOAD DATA #
	#############

	print('... loading data')

	write_cache = False
	use_cache = False

	cache_name = 'APADB'#'22'#'20'#'test2percent'#''#20'#'20_22'

	train_set_split = 0.01
	valid_set_split = 0.01

	#constant_set_split = False
	#constant_valid_set_split = 1000
	#constant_test_set_split = 20000

	rval = None

	if use_cache == False :
	
		#X = numpy.load(dataset + '_input.npy')#_foldprob
		#X = spio.loadmat('apa_fullseq_v_' + dataset + '_input.mat')
		#X = sp.csr_matrix(X["input"])

		loader = numpy.load('npz_apa_seq_' + dataset + '_' + prefix + 'input.npz')
		#loader = numpy.load('npz_apa_fullseq_' + dataset + '_input.npz')
		X = sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape = loader['shape'])


		y = numpy.ravel(numpy.load('apa_' + dataset + '_output.npy')[:,0])
		c = numpy.ravel(numpy.load('apa_' + dataset + '_count.npy'))
		L = numpy.zeros(len(c))
		d = numpy.load('apa_' + dataset + '_distance.npy')
		c_input = numpy.load('apa_' + dataset + '_count.npy')
		s_input = numpy.load('apa_' + dataset + '_sites.npy')

		try :
			proxid = numpy.load('apa_' + dataset + '_proxid.npy')
		except IOError :
			proxid = numpy.zeros(len(y), dtype=object)
			proxid[:] = "DEFAULT.100"

		print(X.shape)
		print(y.shape)
		print(c.shape)
		print(L.shape)
		print(L)
		print(numpy.max(L))

		#L_prune_index = (L != 13) & (L != 14)
		#L_prune_index = (L >= 5) & (L <= 12)
		#L_prune_index = L == 5
		#L_prune_index = L >= 20
		'''X = X[L_prune_index,:]
		y = y[L_prune_index]
		c = c[L_prune_index]
		L = L[L_prune_index]'''

		'''sort_index = numpy.argsort(c)
		X = X[sort_index,:]
		y = y[sort_index]
		c = c[sort_index]
		L = L[sort_index]'''

		#Only non-competing PASes
		'''Floader = numpy.load('npz_apa_seq_' + dataset + '_features.npz')
		F = sp.csr_matrix((Floader['data'], Floader['indices'], Floader['indptr']), shape = Floader['shape'], dtype=numpy.int8)
	
		F_aataaa = numpy.ravel(F[:, 0].todense())
		F_attaaa = numpy.ravel(F[:, 1].todense())
		F_denovo = numpy.ravel(F[:, 5].todense())

		X = X[(F_denovo == 1),:]
		L = L[(F_denovo == 1)]
		y = y[(F_denovo == 1)]
		c = c[(F_denovo == 1)]'''

		
		shuffle_index = None
		'''if shuffle == True :
			shuffle_index = numpy.arange(X.shape[0])
			numpy.random.shuffle(shuffle_index)
			X = X[shuffle_index,:]
			L = L[shuffle_index]
			c = c[shuffle_index]
			y = y[shuffle_index]'''

		if count_filter is not None :
			X = X[c >= count_filter,:]
			L = L[c >= count_filter]
			y = y[c >= count_filter]
			c = c[c >= count_filter]

		constant_set_split = True
		constant_valid_set_split = 1
		constant_test_set_split = X.shape[0] - 2

		print('Done count filtering.')
		print(X.shape)

		if balance_all_libraries == True :

			#List of included libraries
			L_included = [20]#[2, 5, 8, 11]#[22]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[2, 5, 8, 11, 20, 22, 30, 31, 34]#[20]#[2, 5, 8, 11, 20, 22, 30, 31, 32, 33, 34, 35]#[20]#[2, 5, 8, 11, 20]#, 22]#[20, 22]

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
			c = c[arranged_index]
			y = y[arranged_index]

		if balance_libraries == True :
			#1. Get indexes of each library

			apa_fr_index = numpy.nonzero(L < 20)[0]
			apa_sym_prx_index = numpy.nonzero(L >= 20)[0]

			#2. Sort indexes of each library by count

			c_apa_fr = c[apa_fr_index]
			c_apa_sym_prx = c[apa_sym_prx_index]

			sort_index_apa_fr = numpy.argsort(c_apa_fr)
			sort_index_apa_sym_prx = numpy.argsort(c_apa_sym_prx)

			apa_fr_index = apa_fr_index[sort_index_apa_fr]
			apa_sym_prx_index = apa_sym_prx_index[sort_index_apa_sym_prx]

			#3. Shuffle indexes of each library modulo 2

			even_index_apa_fr = numpy.arange(len(apa_fr_index)) % 2 == 0
			odd_index_apa_fr = numpy.arange(len(apa_fr_index)) % 2 == 1

			even_index_apa_sym_prx = numpy.arange(len(apa_sym_prx_index)) % 2 == 0
			odd_index_apa_sym_prx = numpy.arange(len(apa_sym_prx_index)) % 2 == 1

			apa_fr_index_even = apa_fr_index[even_index_apa_fr]
			apa_fr_index_odd = apa_fr_index[odd_index_apa_fr]

			apa_sym_prx_index_even = apa_sym_prx_index[even_index_apa_sym_prx]
			apa_sym_prx_index_odd = apa_sym_prx_index[odd_index_apa_sym_prx]

			apa_fr_index = numpy.concatenate([apa_fr_index_even, apa_fr_index_odd])
			apa_sym_prx_index = numpy.concatenate([apa_sym_prx_index_even, apa_sym_prx_index_odd])

			#4. Join modulo 2

			#arranged_index = []
			#for i in range()


			arranged_index = numpy.zeros(len(apa_fr_index) + len(apa_sym_prx_index), dtype=numpy.int)
			gi = 0
			for i in range(0, len(apa_sym_prx_index) - len(apa_fr_index)) :
				arranged_index[gi] = int(apa_sym_prx_index[i])
				gi += 1
			for i in range(0, len(apa_fr_index)) :
				arranged_index[gi + i * 2] = int(apa_fr_index[i])
			li = 0
			for i in range(len(apa_sym_prx_index) - len(apa_fr_index), len(apa_sym_prx_index)) :
				arranged_index[gi + li * 2 + 1] = int(apa_sym_prx_index[i])
				li += 1

			print('Arranged index:')
			print(len(arranged_index))
			print(arranged_index)

			X = X[arranged_index,:]
			L = L[arranged_index]
			c = c[arranged_index]
			y = y[arranged_index]

		if balance_sublibraries == True :
			#1. Get indexes of each library

			apa_fr_2_index = numpy.nonzero(L == 2)[0]
			apa_fr_5_index = numpy.nonzero(L == 5)[0]
			apa_fr_8_index = numpy.nonzero(L == 8)[0]
			apa_fr_11_index = numpy.nonzero(L == 11)[0]
			apa_fr_20_index = numpy.nonzero(L == 20)[0]
			apa_fr_21_index = numpy.nonzero(L == 21)[0]

			#2. Sort indexes of each library by count

			c_apa_fr_2 = c[apa_fr_2_index]
			c_apa_fr_5 = c[apa_fr_5_index]
			c_apa_fr_8 = c[apa_fr_8_index]
			c_apa_fr_11 = c[apa_fr_11_index]
			c_apa_fr_20 = c[apa_fr_20_index]
			c_apa_fr_21 = c[apa_fr_21_index]

			sort_index_apa_fr_2 = numpy.argsort(c_apa_fr_2)
			sort_index_apa_fr_5 = numpy.argsort(c_apa_fr_5)
			sort_index_apa_fr_8 = numpy.argsort(c_apa_fr_8)
			sort_index_apa_fr_11 = numpy.argsort(c_apa_fr_11)
			sort_index_apa_fr_20 = numpy.argsort(c_apa_fr_20)
			sort_index_apa_fr_21 = numpy.argsort(c_apa_fr_21)

			apa_fr_2_index = apa_fr_2_index[sort_index_apa_fr_2]
			apa_fr_5_index = apa_fr_5_index[sort_index_apa_fr_5]
			apa_fr_8_index = apa_fr_8_index[sort_index_apa_fr_8]
			apa_fr_11_index = apa_fr_11_index[sort_index_apa_fr_11]
			apa_fr_20_index = apa_fr_20_index[sort_index_apa_fr_20]
			apa_fr_21_index = apa_fr_21_index[sort_index_apa_fr_21]

			#3. Shuffle indexes of each library modulo 2

			even_index_apa_fr_2 = numpy.arange(len(apa_fr_2_index)) % 2 == 0
			odd_index_apa_fr_2 = numpy.arange(len(apa_fr_2_index)) % 2 == 1
			even_index_apa_fr_5 = numpy.arange(len(apa_fr_5_index)) % 2 == 0
			odd_index_apa_fr_5 = numpy.arange(len(apa_fr_5_index)) % 2 == 1
			even_index_apa_fr_8 = numpy.arange(len(apa_fr_8_index)) % 2 == 0
			odd_index_apa_fr_8 = numpy.arange(len(apa_fr_8_index)) % 2 == 1
			even_index_apa_fr_11 = numpy.arange(len(apa_fr_11_index)) % 2 == 0
			odd_index_apa_fr_11 = numpy.arange(len(apa_fr_11_index)) % 2 == 1

			even_index_apa_fr_20 = numpy.arange(len(apa_fr_20_index)) % 2 == 0
			odd_index_apa_fr_20 = numpy.arange(len(apa_fr_20_index)) % 2 == 1
			even_index_apa_fr_21 = numpy.arange(len(apa_fr_21_index)) % 2 == 0
			odd_index_apa_fr_21 = numpy.arange(len(apa_fr_21_index)) % 2 == 1

			apa_fr_2_index_even = apa_fr_2_index[even_index_apa_fr_2]
			apa_fr_2_index_odd = apa_fr_2_index[odd_index_apa_fr_2]
			apa_fr_5_index_even = apa_fr_5_index[even_index_apa_fr_5]
			apa_fr_5_index_odd = apa_fr_5_index[odd_index_apa_fr_5]
			apa_fr_8_index_even = apa_fr_8_index[even_index_apa_fr_8]
			apa_fr_8_index_odd = apa_fr_8_index[odd_index_apa_fr_8]
			apa_fr_11_index_even = apa_fr_11_index[even_index_apa_fr_11]
			apa_fr_11_index_odd = apa_fr_11_index[odd_index_apa_fr_11]

			apa_fr_20_index_even = apa_fr_20_index[even_index_apa_fr_20]
			apa_fr_20_index_odd = apa_fr_20_index[odd_index_apa_fr_20]
			apa_fr_21_index_even = apa_fr_21_index[even_index_apa_fr_21]
			apa_fr_21_index_odd = apa_fr_21_index[odd_index_apa_fr_21]

			apa_fr_2_index = numpy.concatenate([apa_fr_2_index_even, apa_fr_2_index_odd])
			apa_fr_5_index = numpy.concatenate([apa_fr_5_index_even, apa_fr_5_index_odd])
			apa_fr_8_index = numpy.concatenate([apa_fr_8_index_even, apa_fr_8_index_odd])
			apa_fr_11_index = numpy.concatenate([apa_fr_11_index_even, apa_fr_11_index_odd])
			apa_fr_20_index = numpy.concatenate([apa_fr_20_index_even, apa_fr_20_index_odd])
			apa_fr_21_index = numpy.concatenate([apa_fr_21_index_even, apa_fr_21_index_odd])

			#4. Join modulo 2

			#arranged_index = []
			#for i in range()


			arranged_index = numpy.zeros(len(apa_fr_2_index) + len(apa_fr_5_index) + len(apa_fr_8_index) + len(apa_fr_11_index) + len(apa_fr_20_index) + len(apa_fr_21_index), dtype=numpy.int)

			min_shuffle = min(min(min(min(min(len(apa_fr_2_index), len(apa_fr_5_index)), len(apa_fr_8_index)), len(apa_fr_11_index)), len(apa_fr_20_index)), len(apa_fr_21_index))

			gi = 0
			for i in range(0, len(apa_fr_2_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_2_index[i])
				gi += 1
			for i in range(0, len(apa_fr_5_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_5_index[i])
				gi += 1
			for i in range(0, len(apa_fr_8_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_8_index[i])
				gi += 1
			for i in range(0, len(apa_fr_11_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_11_index[i])
				gi += 1
			for i in range(0, len(apa_fr_20_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_20_index[i])
				gi += 1
			for i in range(0, len(apa_fr_21_index) - min_shuffle) :
				arranged_index[gi] = int(apa_fr_21_index[i])
				gi += 1

			li = 0
			for i in range(len(apa_fr_2_index) - min_shuffle, len(apa_fr_2_index)) :
				arranged_index[gi + li * 6] = int(apa_fr_2_index[i])
				li += 1
			li = 0
			for i in range(len(apa_fr_5_index) - min_shuffle, len(apa_fr_5_index)) :
				arranged_index[gi + li * 6 + 1] = int(apa_fr_5_index[i])
				li += 1
			li = 0
			for i in range(len(apa_fr_8_index) - min_shuffle, len(apa_fr_8_index)) :
				arranged_index[gi + li * 6 + 2] = int(apa_fr_8_index[i])
				li += 1
			li = 0
			for i in range(len(apa_fr_11_index) - min_shuffle, len(apa_fr_11_index)) :
				arranged_index[gi + li * 6 + 3] = int(apa_fr_11_index[i])
				li += 1
			li = 0
			for i in range(len(apa_fr_20_index) - min_shuffle, len(apa_fr_20_index)) :
				arranged_index[gi + li * 6 + 4] = int(apa_fr_20_index[i])
				li += 1
			li = 0
			for i in range(len(apa_fr_21_index) - min_shuffle, len(apa_fr_21_index)) :
				arranged_index[gi + li * 6 + 5] = int(apa_fr_21_index[i])
				li += 1

			print('Arranged index:')
			print(len(arranged_index))
			print(arranged_index)

			X = X[arranged_index,:]
			L = L[arranged_index]
			c = c[arranged_index]
			y = y[arranged_index]

		if balance_data == True :
			even_index = numpy.arange(len(y)) % 2 == 0
			X_even = X[even_index,:]
			y_even = y[even_index]
			c_even = c[even_index]
			L_even = L[even_index]

			odd_index = numpy.arange(len(y)) % 2 == 1
			X_odd = X[odd_index,:]
			y_odd = y[odd_index]
			c_odd = c[odd_index]
			L_odd = L[odd_index]

			X = sp.csr_matrix(sp.vstack([X_even, X_odd]))
			y = numpy.concatenate([y_even, y_odd])
			c = numpy.concatenate([c_even, c_odd])
			L = numpy.concatenate([L_even, L_odd])


		L_input = numpy.zeros((len(y), 36))
		for i in range(0, len(y)) :
			L_input[i,int(L[i])] = 1

		X_train = None
		X_validation = None
		X_test = None
		y_train = None
		y_validation = None
		y_test = None
		if constant_set_split == False :
			X_train = X[:int(train_set_split * X.shape[0]),:]
			X_validation = X[X_train.shape[0]:X_train.shape[0] + int(valid_set_split * X.shape[0]),:]
			X_test = X[X_train.shape[0] + X_validation.shape[0]:,:]

			y_train = y[:int(train_set_split * y.shape[0])]
			y_validation = y[y_train.shape[0]:y_train.shape[0] + int(valid_set_split * y.shape[0])]
			y_test = y[y_train.shape[0] + y_validation.shape[0]:]
		else :
			X_train = X[:X.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			X_validation = X[X.shape[0] - constant_valid_set_split - constant_test_set_split:X.shape[0] - constant_test_set_split,:]
			X_test = X[X.shape[0] - constant_test_set_split:,:]

			y_train = y[:y.shape[0] - constant_valid_set_split - constant_test_set_split]
			y_validation = y[y.shape[0] - constant_valid_set_split - constant_test_set_split:y.shape[0] - constant_test_set_split]
			y_test = y[y.shape[0] - constant_test_set_split:]
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

		d_train = None
		d_validation = None
		d_test = None
		if constant_set_split == False :
			d_train = d[:int(train_set_split * d.shape[0]),:]
			d_validation = d[d_train.shape[0]:d_train.shape[0] + int(valid_set_split * d.shape[0]),:]
			d_test = d[d_train.shape[0] + d_validation.shape[0]:,:]
		else :
			d_train = d[:d.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			d_validation = d[d.shape[0] - constant_valid_set_split - constant_test_set_split:d.shape[0] - constant_test_set_split,:]
			d_test = d[d.shape[0] - constant_test_set_split:,:]

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

		s_train = None
		s_validation = None
		s_test = None
		if constant_set_split == False :
			s_train = s_input[:int(train_set_split * s_input.shape[0]),:]
			s_validation = s_input[s_train.shape[0]:s_train.shape[0] + int(valid_set_split * s_input.shape[0]),:]
			s_test = s_input[s_train.shape[0] + s_validation.shape[0]:,:]
		else :
			s_train = s_input[:s_input.shape[0] - constant_valid_set_split - constant_test_set_split,:]
			s_validation = s_input[s_input.shape[0] - constant_valid_set_split - constant_test_set_split:s_input.shape[0] - constant_test_set_split,:]
			s_test = s_input[s_input.shape[0] - constant_test_set_split:,:]

		proxid_train = None
		proxid_validation = None
		proxid_test = None
		if constant_set_split == False :
			proxid_train = proxid[:int(train_set_split * proxid.shape[0])]
			proxid_validation = proxid[proxid_train.shape[0]:proxid_train.shape[0] + int(valid_set_split * proxid.shape[0])]
			proxid_test = proxid[proxid_train.shape[0] + proxid_validation.shape[0]:]
		else :
			proxid_train = proxid[:proxid.shape[0] - constant_valid_set_split - constant_test_set_split]
			proxid_validation = proxid[proxid.shape[0] - constant_valid_set_split - constant_test_set_split:proxid.shape[0] - constant_test_set_split]
			proxid_test = proxid[proxid.shape[0] - constant_test_set_split:]

		#Spoof training set
		L_train = numpy.zeros((100, 30))
		d_train = numpy.zeros((100, 1))
		c_train = numpy.zeros((100, 1))
		s_train = numpy.zeros((100, 1))
		proxid_train = numpy.zeros((100, 1))

		train_set_L = shared_dataset(L_train)
		valid_set_L = shared_dataset(L_validation)
		test_set_L = shared_dataset(L_test)

		train_set_d = shared_dataset(d_train)
		valid_set_d = shared_dataset(d_validation)
		test_set_d = shared_dataset(d_test)

		train_set_c = shared_dataset(c_train)
		valid_set_c = shared_dataset(c_validation)
		test_set_c = shared_dataset(c_test)

		train_set_s = shared_dataset(s_train)
		valid_set_s = shared_dataset(s_validation)
		test_set_s = shared_dataset(s_test)

		train_set_proxid = proxid_train
		valid_set_proxid = proxid_validation
		test_set_proxid = proxid_test

		if write_cache == True :
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_train', data=X_train.data, indices=X_train.indices, indptr=X_train.indptr, shape=X_train.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_validation', data=X_validation.data, indices=X_validation.indices, indptr=X_validation.indptr, shape=X_validation.shape)
			numpy.savez('cache_' + cache_name + '_npz_apa_seq_' + dataset + '_input_test', data=X_test.data, indices=X_test.indices, indptr=X_test.indptr, shape=X_test.shape)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_train', L_train)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_validation', L_validation)
			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_L_input_test', L_test)

			numpy.save('cache_' + cache_name + '_apa_' + dataset + '_shuffle_index', shuffle_index)

		
		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None, train_set_d, valid_set_d, test_set_d, train_set_c, valid_set_c, test_set_c, train_set_s, valid_set_s, test_set_s, train_set_proxid, valid_set_proxid, test_set_proxid]
	
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

		train_set_L = theano.shared(L_train, borrow=True)
		valid_set_L = theano.shared(L_validation, borrow=True)
		test_set_L = theano.shared(L_test, borrow=True)

		shuffle_index = numpy.ravel(numpy.load('cache' + cache_name + '_apa_' + dataset + '_shuffle_index.npy'))

		rval = [train_set_x, valid_set_x, test_set_x, shuffle_index, train_set_L, valid_set_L, test_set_L, None]

	return rval

if __name__ == '__main__':
	datasets = load_data()
	#sgd_optimization_mnist()
